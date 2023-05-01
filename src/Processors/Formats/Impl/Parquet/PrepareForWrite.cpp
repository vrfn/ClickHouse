#include "Processors/Formats/Impl/Parquet/Write.h"

#include <Columns/MaskOperations.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnLowCardinality.h>
#include <Columns/ColumnMap.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeMap.h>
#include <DataTypes/DataTypeDateTime64.h>
#include <DataTypes/DataTypeFixedString.h>


/// This file deals with schema conversion and with repetition and definition levels.

/// Schema conversion is pretty straightforward.

/// "Repetition and definition levels" are a somewhat tricky way of encoding information about
/// optional fields and lists.
///
/// If you don't want to learn how these work, feel free to skip the updateRepDefLevels* functions.
/// All you need to know is:
///  * values for nulls are not encoded, so we have to filter nullable columns,
///  * information about all array lengths and nulls is encoded in the arrays `def` and `rep`,
///    which need to be encoded next to the data,
///  * `def` and `rep` arrays can be longer than `primitive_column`, because they include nulls and
///    empty arrays; the values in primitive_column correspond to positions where def[i] == max_def.
///
/// If you do want to learn it, dremel paper: https://research.google/pubs/pub36632/
/// Instead of reading the whole paper, try staring at figures 2-3 for a while - it might be enough.
/// (Why does Parquet do all this instead of just storing array lengths and null masks? I'm not
/// really sure.)
///
/// We calculate the levels recursively, from inner to outer columns.
/// This means scanning the whole array for each Array/Nullable nesting level, which is probably not
/// the most efficient way to do it. But there's usually at most one nesting level, so it's fine.
///
/// Most of this is moot because ClickHouse doesn't support nullable arrays or tuples right now, so
/// almost none of the tricky cases can happen. We implement it in full generality anyway (mostly
/// because I only learned the previous sentence after writing most of the code).


namespace DB::ErrorCodes
{
    extern const int UNKNOWN_TYPE;
    extern const int NOT_IMPLEMENTED;
    extern const int NESTED_TYPE_TOO_DEEP; // I'm 14 and this is deep
    extern const int UNKNOWN_COMPRESSION_METHOD;
}

namespace DB::Parquet
{

/// Thrift structs that Parquet uses for various metadata inside the parquet file.
namespace parq = parquet::format;

namespace
{

void assertNoDefOverflow(ColumnChunkWriteState & s)
{
    if (s.max_def == UINT8_MAX)
        throw Exception(ErrorCodes::NESTED_TYPE_TOO_DEEP,
            "Column has more than 255 levels of nested Array/Nullable. Impressive! Unfortunately, "
            "this is not supported by this Parquet encoder (but is supported by Parquet, if you "
            "really need this for some reason).");
}

void updateRepDefLevelsAndFilterColumnForNullable(ColumnChunkWriteState & s, const NullMap & null_map)
{
    /// Increment definition levels for non-nulls.
    /// Filter the column to contain only non-null values.

    assertNoDefOverflow(s);
    ++s.max_def;

    /// Normal case: no arrays or nullables inside this nullable.
    if (s.max_def == 1)
    {
        chassert(s.def.empty());
        s.def.resize(null_map.size());
        for (size_t i = 0; i < s.def.size(); ++i)
            s.def[i] = !null_map[i];

        /// We could be more efficient with this:
        ///  * Instead of doing the filter() here, we could defer it to writeColumnChunkBody(), at
        ///    least in the simple case of Nullable(Primitive). Then it'll parallelize if the table
        ///    consists of one big tuple.
        ///  * Instead of filtering explicitly, we could build filtering into the data encoder.
        ///  * Instead of filling out the `def` values above, we could point to null_map and build
        ///    the '!' into the encoder.
        /// None of these seem worth the complexity right now.
        s.primitive_column = s.primitive_column->filter(s.def, /*result_size_hint*/ -1);

        return;
    }

    /// Weird general case: Nullable(Array), Nullable(Nullable), or any arbitrary nesting like that.
    /// This is currently not allowed in ClickHouse, but let's support it anyway just in case.

    IColumn::Filter filter;
    size_t row_idx = static_cast<size_t>(-1);
    for (size_t i = 0; i < s.def.size(); ++i)
    {
        row_idx += s.max_rep == 0 || s.rep[i] == 0;
        if (s.def[i] == s.max_def - 1)
            filter.push_back(!null_map[row_idx]);
        s.def[i] += !null_map[row_idx];
    }
    s.primitive_column = s.primitive_column->filter(filter, /*result_size_hint*/ -1);
}

void updateRepDefLevelsForArray(ColumnChunkWriteState & s, const IColumn::Offsets & offsets)
{
    /// Increment all definition levels.
    /// For non-first elements of arrays, increment repetition levels.
    /// For empty arrays, insert a zero into repetition and definition levels arrays.

    assertNoDefOverflow(s);
    ++s.max_def;
    ++s.max_rep;

    /// Common case: no arrays or nullables inside this array.
    if (s.max_rep == 1 && s.max_def == 1)
    {
        s.def.resize_fill(s.primitive_column->size(), 1);
        s.rep.resize_fill(s.primitive_column->size(), 1);
        size_t i = 0;
        for (ssize_t row = 0; row < static_cast<ssize_t>(offsets.size()); ++row)
        {
            size_t n = offsets[row] - offsets[row - 1];
            if (n)
            {
                s.rep[i] = 0;
                i += n;
            }
            else
            {
                s.def.push_back(1);
                s.rep.push_back(1);
                s.def[i] = 0;
                s.rep[i] = 0;
                i += 1;
            }
        }
        return;
    }

    /// General case: Array(Array), Array(Nullable), or any arbitrary nesting like that.

    for (auto & x : s.def)
        ++x;

    if (s.max_rep == 1)
        s.rep.resize_fill(s.def.size(), 1);
    else
        for (auto & x : s.rep)
            ++x;

    PaddedPODArray<UInt8> mask(s.def.size(), 1); // for inserting zeroes to rep and def
    size_t i = 0; // in the input (s.def/s.rep)
    size_t empty_arrays = 0;
    for (ssize_t row = 0; row < static_cast<ssize_t>(offsets.size()); ++row)
    {
        size_t n = offsets[row] - offsets[row - 1];
        if (n)
        {
            /// Un-increment the first rep of the array.
            /// Skip n "items" in the nested column; first element of each item has rep = 1
            /// (we incremented it above).
            chassert(s.rep[i] == 1);
            --s.rep[i];
            do
            {
                ++i;
                if (i == s.rep.size())
                {
                    --n;
                    chassert(n == 0);
                    break;
                }
                n -= s.rep[i] == 1;
            } while (n);
        }
        else
        {
            mask.push_back(1);
            mask[i + empty_arrays] = 0;
            ++empty_arrays;
        }
    }

    if (empty_arrays != 0)
    {
        expandDataByMask(s.def, mask, false);
        expandDataByMask(s.rep, mask, false);
    }
}

parq::CompressionCodec::type compressionMethodToParquet(CompressionMethod c)
{
    switch (c)
    {
        case CompressionMethod::None: return parq::CompressionCodec::UNCOMPRESSED;
        case CompressionMethod::Snappy: return parq::CompressionCodec::SNAPPY;
        case CompressionMethod::Gzip: return parq::CompressionCodec::GZIP;
        case CompressionMethod::Brotli: return parq::CompressionCodec::BROTLI;
        case CompressionMethod::Lz4: return parq::CompressionCodec::LZ4_RAW;
        case CompressionMethod::Zstd: return parq::CompressionCodec::ZSTD;

        default:
            throw Exception(ErrorCodes::UNKNOWN_COMPRESSION_METHOD, "Compression method {} is not supported by Parquet", toContentEncodingName(c));
    }
}

/// Depth-first traversal of the schema tree for this column.
void prepareColumnRecursive(
    ColumnPtr column, DataTypePtr type, const std::string & name, const WriteOptions & options,
    ColumnChunkWriteStates & states, SchemaElements & schemas);

void preparePrimitiveColumn(ColumnPtr column, DataTypePtr type, const std::string & name,
    const WriteOptions & options, ColumnChunkWriteStates & states, SchemaElements & schemas)
{
    /// Add physical column info.
    auto & state = states.emplace_back();
    state.primitive_column = column;
    state.compression = options.compression;

    state.column_chunk.__isset.meta_data = true;
    state.column_chunk.meta_data.__set_path_in_schema({name});
    state.column_chunk.meta_data.__set_codec(compressionMethodToParquet(state.compression));

    /// Add logical schema leaf.
    auto & schema = schemas.emplace_back();
    schema.__set_repetition_type(parq::FieldRepetitionType::REQUIRED);
    schema.__set_name(name);

    /// Convert the type enums.

    using T = parq::Type;
    using C = parq::ConvertedType;

    auto types = [&](T::type type_, std::optional<C::type> converted, parq::LogicalType logical)
    {
        state.column_chunk.meta_data.__set_type(type_);
        schema.__set_type(type_);
        if (converted)
            schema.__set_converted_type(*converted);
        schema.__set_logicalType(logical);
    };

    auto int_type = [](Int8 bits, bool signed_)
    {
        parq::LogicalType t;
        t.__isset.INTEGER = true;
        t.INTEGER.__set_bitWidth(bits);
        t.INTEGER.__set_isSigned(signed_);
        return t;
    };

    // TODO: remember to assign type_length, scale, precision

    switch (type->getTypeId())
    {
        case TypeIndex::UInt8:  types(T::INT32, C::UINT_8 , int_type(8 , false)); break;
        case TypeIndex::UInt16: types(T::INT32, C::UINT_16, int_type(16, false)); break;
        case TypeIndex::UInt32: types(T::INT32, C::UINT_32, int_type(32, false)); break;
        case TypeIndex::UInt64: types(T::INT64, C::UINT_64, int_type(64, false)); break;
        case TypeIndex::Int8:   types(T::INT32, C::INT_8  , int_type(8 , true )); break;
        case TypeIndex::Int16:  types(T::INT32, C::INT_16 , int_type(16, true )); break;
        case TypeIndex::Int32:  types(T::INT32, C::INT_32 , int_type(32, true )); break;
        case TypeIndex::Int64:  types(T::INT64, C::INT_64 , int_type(64, true )); break;

        case TypeIndex::String:
        {
            parq::LogicalType t;
            t.__set_STRING({});
            types(T::BYTE_ARRAY, C::UTF8, t);
            break;
        }

        case TypeIndex::UInt128:
        case TypeIndex::UInt256:
        case TypeIndex::Int128:
        case TypeIndex::Int256:

        case TypeIndex::Float32:
        case TypeIndex::Float64:

        case TypeIndex::Date:
        case TypeIndex::Date32:
        case TypeIndex::DateTime:
        case TypeIndex::DateTime64:

        case TypeIndex::FixedString:

        case TypeIndex::Enum8:
        case TypeIndex::Enum16:

        case TypeIndex::Decimal32:
        case TypeIndex::Decimal64:
        case TypeIndex::Decimal128:
        case TypeIndex::Decimal256:

        case TypeIndex::UUID:

        case TypeIndex::LowCardinality:

        case TypeIndex::IPv4:
        case TypeIndex::IPv6:
            /// TODO
            throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Data type {} not implemented", type->getFamilyName());

        default:
            throw Exception(ErrorCodes::UNKNOWN_TYPE, "Internal type '{}' of column '{}' is not supported for conversion into Parquet data format.", type->getFamilyName(), name);
    }
}

void prepareColumnArray(
    ColumnPtr column, DataTypePtr type, const std::string & name, const WriteOptions & options,
    ColumnChunkWriteStates & states, SchemaElements & schemas)
{
    const auto * column_array = assert_cast<const ColumnArray *>(column.get());
    ColumnPtr nested_column = column_array->getDataPtr();
    DataTypePtr nested_type = assert_cast<const DataTypeArray *>(type.get())->getNestedType();
    const auto & offsets = column_array->getOffsets();

    /// We mimic the schema that arrow produces for lists, which is:
    ///
    /// required group `name` (List):
    ///   repeated group "list":
    ///     <recurse into nested type> "item"
    ///
    /// If nested type is not Nullable, we could remove one unnecessary level:
    ///
    /// required group `name` (List):
    ///   repeated <recurse into nested type> "item"
    ///
    /// But we don't do that in case it breaks compatibility with something.

    /// Add the groups schema.

    schemas.emplace_back();
    schemas.emplace_back();
    auto & list_schema = schemas[schemas.size() - 2];
    auto & item_schema = schemas[schemas.size() - 1];

    list_schema.__set_repetition_type(parq::FieldRepetitionType::REQUIRED);
    list_schema.__set_name(name);
    list_schema.__set_num_children(1);
    list_schema.__set_converted_type(parq::ConvertedType::LIST);
    list_schema.__isset.logicalType = true;
    list_schema.logicalType.__set_LIST({});

    item_schema.__set_repetition_type(parq::FieldRepetitionType::REPEATED);
    item_schema.__set_name("list");
    item_schema.__set_num_children(1);

    std::array<std::string, 2> path_prefix = {list_schema.name, item_schema.name};
    size_t child_states_begin = states.size();

    /// Recurse.
    prepareColumnRecursive(nested_column, nested_type, "item", options, states, schemas);

    /// Update repetition+definition levels and fully-qualified column names (x -> myarray.list.x).
    for (size_t i = child_states_begin; i < states.size(); ++i)
    {
        Strings & path = states[i].column_chunk.meta_data.path_in_schema;
        /// O(nesting_depth^2), but who cares.
        path.insert(path.begin(), path_prefix.begin(), path_prefix.end());

        updateRepDefLevelsForArray(states[i], offsets);
    }
}

void prepareColumnNullable(
    ColumnPtr column, DataTypePtr type, const std::string & name, const WriteOptions & options,
    ColumnChunkWriteStates & states, SchemaElements & schemas)
{
    const ColumnNullable * column_nullable = assert_cast<const ColumnNullable *>(column.get());
    ColumnPtr nested_column = column_nullable->getNestedColumnPtr();
    DataTypePtr nested_type = assert_cast<const DataTypeNullable *>(type.get())->getNestedType();
    const NullMap & null_map = column_nullable->getNullMapData();

    size_t child_states_begin = states.size();
    size_t child_schema_idx = schemas.size();

    prepareColumnRecursive(nested_column, nested_type, name, options, states, schemas);

    if (schemas[child_schema_idx].repetition_type == parq::FieldRepetitionType::REQUIRED)
    {
        /// Normal case: we just slap a FieldRepetitionType::OPTIONAL onto the nested column.
        schemas[child_schema_idx].repetition_type = parq::FieldRepetitionType::OPTIONAL;
    }
    else
    {
        /// Weird case: Nullable(Nullable(...)).
        /// This is probably not allowed in ClickHouse, but let's support it just in case.
        auto & schema = *schemas.insert(schemas.begin() + child_schema_idx, {});
        schema.__set_repetition_type(parq::FieldRepetitionType::OPTIONAL);
        schema.__set_name("nullable");
        schema.__set_num_children(1);
        for (size_t i = child_states_begin; i < states.size(); ++i)
        {
            Strings & path = states[i].column_chunk.meta_data.path_in_schema;
            path.insert(path.begin(), schema.name + ".");
        }
    }

    for (size_t i = child_states_begin; i < states.size(); ++i)
    {
        auto & s = states[i];
        updateRepDefLevelsAndFilterColumnForNullable(s, null_map);
    }
}

void prepareColumnRecursive(
    ColumnPtr column, DataTypePtr type, const std::string & name, const WriteOptions & options,
    ColumnChunkWriteStates & states, SchemaElements & schemas)
{
    switch (type->getTypeId())
    {
        case TypeIndex::Array:
            prepareColumnArray(column, type, name, options, states, schemas);
            break;
        case TypeIndex::Nullable:
            prepareColumnNullable(column, type, name, options, states, schemas);
            break;
        case TypeIndex::Tuple:
        case TypeIndex::Map:
            /// TODO
            throw Exception(ErrorCodes::NOT_IMPLEMENTED, "not implemented");
        default:
            preparePrimitiveColumn(column, type, name, options, states, schemas);
            break;
    }
}

}

SchemaElements convertSchema(const Block & sample, const WriteOptions & options)
{
    SchemaElements schema;
    auto & root = schema.emplace_back();
    root.__set_name("schema");
    root.__set_num_children(static_cast<Int32>(sample.columns()));

    for (auto & c : sample)
        prepareColumnForWrite(c.column, c.type, c.name, options, nullptr, &schema);

    return schema;
}

void prepareColumnForWrite(
    ColumnPtr column, DataTypePtr type, const std::string & name, const WriteOptions & options,
    ColumnChunkWriteStates * out_columns_to_write, SchemaElements * out_schema)
{
    if (column->size() == 0 && out_columns_to_write != nullptr)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Empty column passed to Parquet encoder");

    /// TODO: Handle ColumnLowCardinality natively. Maybe get rid of the normal dictionary encoding
    ///       implementation and just convert all columns to LowCardinality here (with fallback to
    ///       dense if the dictionary gets big).
    type = recursiveRemoveLowCardinality(type);
    column = recursiveRemoveLowCardinality(column);

    ColumnChunkWriteStates states;
    SchemaElements schemas;
    prepareColumnRecursive(column, type, name, options, states, schemas);

    if (out_columns_to_write)
        for (auto & s : states)
            out_columns_to_write->push_back(std::move(s));
    if (out_schema)
        out_schema->insert(out_schema->end(), schemas.begin(), schemas.end());

    if (column->empty())
        states.clear();
}

}
