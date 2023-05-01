#include "Processors/Formats/Impl/Parquet/Write.h"
#include "Processors/Formats/Impl/Parquet/ThriftUtil.h"
#include <parquet/column_writer.h>
#include <parquet/encoding.h>
#include <lz4.h>
#include <Columns/MaskOperations.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnLowCardinality.h>
#include <Columns/ColumnMap.h>
#include <IO/WriteHelpers.h>
#include "config_version.h"

namespace DB::ErrorCodes
{
    extern const int UNKNOWN_TYPE;
    extern const int NOT_IMPLEMENTED;
    extern const int CANNOT_COMPRESS;
    extern const int LIMIT_EXCEEDED;
}

namespace DB::Parquet
{

namespace parq = parquet::format;

namespace
{

/// Returns either `source` or `scratch`.
PODArray<char> & compress(PODArray<char> & source, PODArray<char> & scratch, CompressionMethod method)
{
    /// We could use wrapWriteBufferWithCompressionMethod() for everything, but I worry about the
    /// overhead of creating a bunch of WriteBuffers on each page (thousands of values).
    switch (method)
    {
        case CompressionMethod::None:
            return source;

        case CompressionMethod::Lz4:
        {
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wold-style-cast"

            size_t max_dest_size = LZ4_COMPRESSBOUND(source.size());

            #pragma clang diagnostic pop

            if (max_dest_size > std::numeric_limits<int>::max())
                throw Exception(ErrorCodes::CANNOT_COMPRESS, "Cannot compress column of size {}", formatReadableSizeWithBinarySuffix(source.size()));

            scratch.resize(max_dest_size);

            int compressed_size = LZ4_compress_default(
                source.data(),
                scratch.data(),
                static_cast<int>(source.size()),
                static_cast<int>(max_dest_size));

            scratch.resize(static_cast<size_t>(compressed_size));
            return scratch;
        }

        default:
        {
            auto dest_buf = std::make_unique<WriteBufferFromVector<PODArray<char>>>(scratch);
            auto compressed_buf = wrapWriteBufferWithCompressionMethod(
                std::move(dest_buf),
                method,
                /*level*/ 3,
                source.size(),
                /*existing_memory*/ source.data());
            chassert(compressed_buf->position() == source.data());
            chassert(compressed_buf->available() == source.size());
            compressed_buf->position() += source.size();
            compressed_buf->finalize();
            return scratch;
        }
    }
}

void encodeRepDefLevelsRLE(const UInt8 * data, size_t size, UInt8 max_level, PODArray<Int16> & scratch, PODArray<char> & out)
{
    /// Reusing arrow code for RLE encoding.
    /// TODO: We unnecessarily convert our UInt8 array to Int16 array, which LevelEncoder then
    ///       immediately converts to UInt64. We should skip LevelEncoder and use
    ///       parquet::util::RleEncoder directly. LevelEncoder adds ~no value.

    scratch.resize(size);
    for (size_t i = 0; i < size; ++i)
        scratch[i] = static_cast<Int16>(data[i]);

    size_t offset = out.size();
    size_t prefix_size = sizeof(Int32);
    int max_rle_size = parquet::LevelEncoder::MaxBufferSize(
        parquet::Encoding::RLE, max_level, static_cast<int>(size));

    out.resize(offset + prefix_size + max_rle_size);

    parquet::LevelEncoder encoder;
    encoder.Init(parquet::Encoding::RLE, max_level, static_cast<int>(size), reinterpret_cast<uint8_t *>(out.data() + offset + prefix_size), max_rle_size);
    int encoded = encoder.Encode(static_cast<int>(size), scratch.data());
    chassert(encoded == size);

    Int32 len = encoder.len();
    memcpy(out.data() + offset, &len, sizeof(Int32));

    out.resize(offset + prefix_size + encoder.len());
}

/// The column usually needs to be converted to one of Parquet physical types, e.g. UInt16 -> Int32
/// or [element of ColumnString] -> std::string_view.
/// We do this conversion in small batches rather than all at once, just before encoding the batch,
/// in hopes of getting better performance through cache locality.
/// The Coverter* structs below are responsible for that.
/// When conversion is not needed, getBatch() will just return pointer into original data.

template <typename From, typename To>
struct ConverterNumeric
{
    const From * data;
    PODArray<To> buf;

    ConverterNumeric(const ColumnPtr & c)
        : data(assert_cast<const ColumnVector<From> &>(*c).getData().data()) {}

    const To * getBatch(size_t offset, size_t count)
    {
        if constexpr (sizeof(From) == sizeof(To))
            return reinterpret_cast<const To *>(data + offset);
        else
        {
            buf.resize(count);
            for (size_t i = 0; i < count; ++i)
                buf[i] = static_cast<To>(data[offset + i]);
            return buf.data();
        }
    }
};

struct ConverterString
{
    const ColumnString & column;
    PODArray<parquet::ByteArray> buf;

    ConverterString(const ColumnPtr & c) : column(assert_cast<const ColumnString &>(*c)) {}

    const parquet::ByteArray * getBatch(size_t offset, size_t count)
    {
        buf.resize(count);
        for (size_t i = 0; i < count; ++i)
        {
            StringRef s = column.getDataAt(offset + i);
            buf[i] = parquet::ByteArray(static_cast<UInt32>(s.size), reinterpret_cast<const uint8_t *>(s.data));
        }
        return buf.data();
    }
};

void addToEncodingsUsed(ColumnChunkWriteState & s, parq::Encoding::type e)
{
    if (!std::count(s.column_chunk.meta_data.encodings.begin(), s.column_chunk.meta_data.encodings.end(), e))
        s.column_chunk.meta_data.encodings.push_back(e);
}

/// Reused across pages to reduce number of allocations and improve locality.
struct PageScratchBuffers
{
    PODArray<char> encoded;
    PODArray<char> compressed;
    PODArray<Int16> levels;
};

void writePage(const parq::PageHeader & header, const PODArray<char> & compressed, ColumnChunkWriteState & s, WriteBuffer & out)
{
    size_t header_size = serializeThriftStruct(header, out);
    out.write(compressed.data(), compressed.size());

    /// Remember first data page and first dictionary page.
    if (header.__isset.data_page_header && s.column_chunk.meta_data.data_page_offset == -1)
        s.column_chunk.meta_data.__set_data_page_offset(s.column_chunk.meta_data.total_compressed_size);
    if (header.__isset.dictionary_page_header && !s.column_chunk.meta_data.__isset.dictionary_page_offset)
        s.column_chunk.meta_data.__set_dictionary_page_offset(s.column_chunk.meta_data.total_compressed_size);

    s.column_chunk.meta_data.total_uncompressed_size += header.uncompressed_page_size + header_size;
    s.column_chunk.meta_data.total_compressed_size += header.compressed_page_size + header_size;
}

template <typename ParquetDType, typename Converter>
void writeColumnUsingParquetEncoder(
    ColumnChunkWriteState & s, const WriteOptions & options, WriteBuffer & out, Converter && converter)
{
    size_t num_values = s.max_def > 0 ? s.def.size() : s.primitive_column->size();
    auto encoding = options.encoding;

    /// We start with dictionary encoding, then switch to `encoding` (non-dictionary) if the
    /// dictionary gets too big. That's how arrow does it.
    bool initially_used_dictionary = options.use_dictionary_encoding;
    bool currently_using_dictionary = initially_used_dictionary;

    /// Could use an arena here (by passing a custom MemoryPool), to reuse memory across pages.
    auto encoder = parquet::MakeTypedEncoder<ParquetDType>(
        static_cast<parquet::Encoding::type>(encoding) /* ignored if using dictionary */, currently_using_dictionary);

    struct PageData
    {
        parq::PageHeader header;
        PODArray<char> data;
    };
    std::vector<PageData> dict_encoded_pages; // can't write them out until we have full dictionary

    /// TODO: dictionary encoding, fallback to non-dict if dictionary gets big;
    ///       don't forget to assign dictionary_page_offset

    PageScratchBuffers scratch;
    /// Start of current page.
    size_t def_offset = 0; // index in def and rep
    size_t data_offset = 0; // index in primitive_column

    auto flush_page = [&](size_t def_count, size_t data_count){
        auto & encoded = scratch.encoded;
        encoded.clear();

        /// Concatenate encoded rep, def, and data.

        if (s.max_rep > 0)
            encodeRepDefLevelsRLE(s.rep.data() + def_offset, def_count, s.max_rep, scratch.levels, encoded);
        if (s.max_def > 0)
            encodeRepDefLevelsRLE(s.def.data() + def_offset, def_count, s.max_def, scratch.levels, encoded);

        std::shared_ptr<parquet::Buffer> values = encoder->FlushValues(); // resets it for next page

        encoded.resize(encoded.size() + values->size());
        memcpy(encoded.data() + encoded.size() - values->size(), values->data(), values->size());
        values.reset();

        if (encoded.size() > INT32_MAX)
            throw Exception(ErrorCodes::CANNOT_COMPRESS, "Uncompressed page is too big: {}", encoded.size());

        size_t uncompressed_size = encoded.size();
        auto & compressed = compress(encoded, scratch.compressed, s.compression);

        if (compressed.size() > INT32_MAX)
            throw Exception(ErrorCodes::CANNOT_COMPRESS, "Compressed page is too big: {}", compressed.size());

        parq::PageHeader header;
        header.__set_type(parq::PageType::DATA_PAGE);
        header.__set_uncompressed_page_size(static_cast<int>(uncompressed_size));
        header.__set_compressed_page_size(static_cast<int>(compressed.size()));
        header.__isset.data_page_header = true;
        auto & d = header.data_page_header;
        d.__set_num_values(static_cast<Int32>(def_count));
        d.__set_encoding(currently_using_dictionary ? parq::Encoding::RLE_DICTIONARY : encoding);
        d.__set_definition_level_encoding(parq::Encoding::RLE);
        d.__set_repetition_level_encoding(parq::Encoding::RLE);
        /// We could also put checksum in `header.crc`, but apparently no one uses it:
        /// https://issues.apache.org/jira/browse/PARQUET-594

        if (currently_using_dictionary)
        {
            dict_encoded_pages.push_back({.header = std::move(header)});
            std::swap(dict_encoded_pages.back().data, compressed);
        }
        else
        {
            writePage(header, compressed, s, out);
        }

        def_offset += def_count;
        data_offset += data_count;
    };

    auto flush_dict_if_needed = [&](bool only_if_too_big) -> bool
    {
        if (!currently_using_dictionary)
            return false;

        auto * dict_encoder = dynamic_cast<parquet::DictEncoder<ParquetDType> *>(encoder.get());

        int dict_size = dict_encoder->dict_encoded_size();
        if (only_if_too_big && static_cast<size_t>(dict_size) < options.dictionary_size_limit)
            return false;

        auto & encoded = scratch.encoded;
        encoded.resize(static_cast<size_t>(dict_size));
        dict_encoder->WriteDict(reinterpret_cast<uint8_t *>(encoded.data()));

        auto & compressed = compress(encoded, scratch.compressed, s.compression);

        if (compressed.size() > INT32_MAX)
            throw Exception(ErrorCodes::CANNOT_COMPRESS, "Compressed dictionary page is too big: {}", compressed.size());

        parq::PageHeader header;
        header.__set_type(parq::PageType::DICTIONARY_PAGE);
        header.__set_uncompressed_page_size(dict_size);
        header.__set_compressed_page_size(static_cast<int>(compressed.size()));
        header.__isset.dictionary_page_header = true;
        header.dictionary_page_header.__set_num_values(dict_encoder->num_entries());
        header.dictionary_page_header.__set_encoding(parq::Encoding::PLAIN);

        writePage(header, compressed, s, out);

        for (auto & p : dict_encoded_pages)
            writePage(p.header, p.data, s, out);

        dict_encoded_pages.clear();
        encoder.reset();

        return true;
    };

    while (def_offset < num_values)
    {
        if (flush_dict_if_needed(true))
        {
            /// Fallback to non-dictionary encoding.
            currently_using_dictionary = false;
            encoder = parquet::MakeTypedEncoder<ParquetDType>(
                static_cast<parquet::Encoding::type>(encoding));
        }

        /// Pick enough data for a page.
        size_t next_def_offset = def_offset;
        size_t next_data_offset = data_offset;
        do
        {
            /// Bite off a batch of defs and corresponding data values.
            size_t def_count = std::min(options.write_batch_size, num_values - next_def_offset);
            size_t data_count = 0;
            if (s.max_def == 0)
                data_count = def_count;
            else
                for (size_t i = 0; i < def_count; ++i)
                    data_count += s.def[next_def_offset + i] == s.max_def;

            /// Encode the data (but not the levels yet), so that we can estimate its encoded size.
            const typename ParquetDType::c_type * converted = converter.getBatch(next_data_offset, data_count);
            encoder->Put(converted, static_cast<int>(data_count));

            /// TODO: Calculate statistics, probably using parquet/statistics.h
            ///       Put them in page headers and column chunk metadata.

            next_def_offset += def_count;
            next_data_offset += data_count;
        }
        while (next_def_offset < num_values &&
               static_cast<size_t>(encoder->EstimatedDataEncodedSize()) < options.data_page_size);

        flush_page(next_def_offset - def_offset, next_data_offset - data_offset);
    }

    flush_dict_if_needed(false);

    chassert(data_offset == s.primitive_column->size());

    /// Report which encodings we've used.
    if (s.max_rep > 0 || s.max_def > 0)
        addToEncodingsUsed(s, parq::Encoding::RLE); // levels
    if (!currently_using_dictionary)
        addToEncodingsUsed(s, encoding); // non-dictionary encoding
    if (initially_used_dictionary)
    {
        addToEncodingsUsed(s, parq::Encoding::PLAIN); // dictionary itself
        addToEncodingsUsed(s, parq::Encoding::RLE_DICTIONARY); // ids
    }
}

}

void writeFileHeader(WriteBuffer & out)
{
    /// Write the magic bytes. We're a wizard now.
    out.write("PAR1", 4);
}

void writeColumnChunkBody(ColumnChunkWriteState & s, const WriteOptions & options, WriteBuffer & out)
{
    s.column_chunk.meta_data.__set_num_values(s.max_def > 0 ? s.def.size() : s.primitive_column->size());

    /// We'll be updating these as we go.
    s.column_chunk.meta_data.__set_encodings({});
    s.column_chunk.meta_data.__set_total_compressed_size(0);
    s.column_chunk.meta_data.__set_total_uncompressed_size(0);
    s.column_chunk.meta_data.__set_data_page_offset(-1);

    // TODO: statistics

    switch (s.primitive_column->getDataType())
    {
        /// Numeric conversion to Int32 or Int64.
        #define N(source_type, parquet_dtype) \
            writeColumnUsingParquetEncoder<parquet::parquet_dtype>( \
                s, options, out, ConverterNumeric<source_type, parquet::parquet_dtype::c_type>( \
                    s.primitive_column))
        case TypeIndex::UInt8  : N(UInt8  , Int32Type ); break;
        case TypeIndex::UInt16 : N(UInt16 , Int32Type ); break;
        case TypeIndex::UInt32 : N(UInt32 , Int32Type ); break;
        case TypeIndex::UInt64 : N(UInt64 , Int64Type ); break;
        case TypeIndex::Int8   : N(Int8   , Int32Type ); break;
        case TypeIndex::Int16  : N(Int16  , Int32Type ); break;
        case TypeIndex::Int32  : N(Int32  , Int32Type ); break;
        case TypeIndex::Int64  : N(Int64  , Int64Type ); break;
        case TypeIndex::Float32: N(Float32, FloatType ); break;
        case TypeIndex::Float64: N(Float32, DoubleType); break;
        #undef N

        case TypeIndex::String:
            writeColumnUsingParquetEncoder<parquet::ByteArrayType>(
                s, options, out, ConverterString(s.primitive_column));
            break;

        case TypeIndex::UInt128:
        case TypeIndex::UInt256:
        case TypeIndex::Int128:
        case TypeIndex::Int256:
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
        case TypeIndex::Nullable:
        case TypeIndex::LowCardinality:
        case TypeIndex::IPv4:
        case TypeIndex::IPv6:
            /// TODO
            throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Type {} not implemented yet", s.primitive_column->getFamilyName());

        default:
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Unexpected column type: {}", s.primitive_column->getFamilyName());
    }

    /// Free some memory.
    s.primitive_column = {};
    s.def = {};
    s.rep = {};
}

parq::ColumnChunk finalizeColumnChunkAndWriteFooter(
    size_t offset_in_file, ColumnChunkWriteState s, const WriteOptions &, WriteBuffer & out)
{
    if (s.column_chunk.meta_data.data_page_offset != -1)
        s.column_chunk.meta_data.data_page_offset += offset_in_file;
    if (s.column_chunk.meta_data.__isset.dictionary_page_offset)
        s.column_chunk.meta_data.dictionary_page_offset += offset_in_file;
    s.column_chunk.file_offset = offset_in_file + s.column_chunk.meta_data.total_compressed_size;

    serializeThriftStruct(s.column_chunk, out);

    return std::move(s.column_chunk);
}

parq::RowGroup makeRowGroup(std::vector<parq::ColumnChunk> column_chunks, size_t num_rows)
{
    parq::RowGroup r;
    r.num_rows = num_rows;
    r.columns = std::move(column_chunks);
    for (auto & c : r.columns)
        r.total_byte_size += c.meta_data.total_uncompressed_size;
    return r;
}

void writeFileFooter(std::vector<parq::RowGroup> row_groups, SchemaElements schema, WriteBuffer & out)
{
    parq::FileMetaData meta;
    meta.version = 2;
    meta.schema = std::move(schema);
    meta.row_groups = std::move(row_groups);
    for (auto & r : meta.row_groups)
        meta.num_rows += r.num_rows;
    meta.__set_created_by(VERSION_NAME " " VERSION_DESCRIBE);

    size_t footer_size = serializeThriftStruct(meta, out);

    if (footer_size > INT32_MAX)
        throw Exception(ErrorCodes::LIMIT_EXCEEDED, "Parquet file metadata too big: {}", footer_size);

    writeIntBinary(static_cast<int>(footer_size), out);
    out.write("PAR1", 4);
}

}
