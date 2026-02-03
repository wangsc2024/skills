# Qpdf - Advanced

**Pages:** 4

---

## Linearization

**URL:** https://qpdf.readthedocs.io/en/stable/linearization.html

**Contents:**
- Linearization
- Basic Strategy for Linearization
- Preparing For Linearization
- Optimization
- Writing Linearized Files
- Calculating Linearization Data
- Known Issues with Linearization
- Debugging Note

This chapter describes how QPDF and QPDFWriter implement creation and processing of linearized PDFs.

To avoid the incestuous problem of having the qpdf library validate its own linearized files, we have a special linearized file checking mode which can be invoked via qpdf --check-linearization (or qpdf --check). This mode reads the linearization parameter dictionary and the hint streams and validates that object ordering, parameters, and hint stream contents are correct. The validation code was first tested against linearized files created by external tools (Acrobat and pdlin) and then used to validate files created by QPDFWriter itself.

Before creating a linearized PDF file from any other PDF file, the PDF file must be altered such that all page attributes are propagated down to the page level (and not inherited from parents in the /Pages tree). We also have to know which objects refer to which other objects, being concerned with page boundaries and a few other cases. We refer to this part of preparing the PDF file as optimization, discussed in Optimization. Note the, in this context, the term optimization is a qpdf term, and the term linearization is a term from the PDF specification. Do not be confused by the fact that many applications refer to linearization as optimization or web optimization.

When creating linearized PDF files from optimized PDF files, there are really only a few issues that need to be dealt with:

Creation of hints tables

Placing objects in the correct order

Filling in offsets and byte sizes

In order to perform various operations such as linearization and splitting files into pages, it is necessary to know which objects are referenced by which pages, page thumbnails, and root and trailer dictionary keys. It is also necessary to ensure that all page-level attributes appear directly at the page level and are not inherited from parents in the pages tree.

We refer to the process of enforcing these constraints as optimization. As mentioned above, note that some applications refer to linearization as optimization. Although this optimization was initially motivated by the need to create linearized files, we are using these terms separately.

PDF file optimization is implemented in the QPDF_optimization.cc source file. That file is richly commented and serves as the primary reference for the optimization process.

After optimization has been completed, the private member variables obj_user_to_objects and object_to_obj_users in QPDF have been populated. Any object that has more than one value in the object_to_obj_users table is shared. Any object that has exactly one value in the object_to_obj_users table is private. To find all the private objects in a page or a trailer or root dictionary key, one merely has make this determination for each element in the obj_user_to_objects table for the given page or key.

Note that pages and thumbnails have different object user types, so the above test on a page will not include objects referenced by the page’s thumbnail dictionary and nothing else.

We will create files with only primary hint streams. We will never write overflow hint streams. (As of PDF version 1.4, Acrobat doesn’t either, and they are never necessary.) The hint streams contain offset information to objects that point to where they would be if the hint stream were not present. This means that we have to calculate all object positions before we can generate and write the hint table. This means that we have to generate the file in two passes. To make this reliable, QPDFWriter in linearization mode invokes exactly the same code twice to write the file to a pipeline.

In the first pass, the target pipeline is a count pipeline chained to a discard pipeline. The count pipeline simply passes its data through to the next pipeline in the chain but can return the number of bytes passed through it at any intermediate point. The discard pipeline is an end of line pipeline that just throws its data away. The hint stream is not written and dummy values with adequate padding are stored in the first cross reference table, linearization parameter dictionary, and /Prev key of the first trailer dictionary. All the offset, length, object renumbering information, and anything else we need for the second pass is stored.

At the end of the first pass, this information is passed to the QPDF class which constructs a compressed hint stream in a memory buffer and returns it. QPDFWriter uses this information to write a complete hint stream object into a memory buffer. At this point, the length of the hint stream is known.

In the second pass, the end of the pipeline chain is a regular file instead of a discard pipeline, and we have known values for all the offsets and lengths that we didn’t have in the first pass. We have to adjust offsets that appear after the start of the hint stream by the length of the hint stream, which is known. Anything that is of variable length is padded, with the padding code surrounding any writing code that differs in the two passes. This ensures that changes to the way things are represented never results in offsets that were gathered during the first pass becoming incorrect for the second pass.

Using this strategy, we can write linearized files to a non-seekable output stream with only a single pass to disk or wherever the output is going.

Once a file is optimized, we have information about which objects access which other objects. We can then process these tables to decide which part (as described in “Linearized PDF Document Structure” in the PDF specification) each object is contained within. This tells us the exact order in which objects are written. The QPDFWriter class asks for this information and enqueues objects for writing in the proper order. It also turns on a check that causes an exception to be thrown if an object is encountered that has not already been queued. (This could happen only if there were a bug in the traversal code used to calculate the linearization data.)

There are a handful of known issues with this linearization code. These issues do not appear to impact the behavior of linearized files which still work as intended: it is possible for a web browser to begin to display them before they are fully downloaded. In fact, it seems that various other programs that create linearized files have many of these same issues. These items make reference to terminology used in the linearization appendix of the PDF specification.

Thread Dictionary information keys appear in part 4 with the rest of Threads instead of in part 9. Objects in part 9 are not grouped together functionally.

We are not calculating numerators for shared object positions within content streams or interleaving them within content streams.

We generate only page offset, shared object, and outline hint tables. It would be relatively easy to add some additional tables. We gather most of the information needed to create thumbnail hint tables. There are comments in the code about this.

The qpdf --show-linearization command can show the complete contents of linearization hint streams. To look at the raw data, you can extract the filtered contents of the linearization hint tables using qpdf --show-object=n --filtered-stream-data. Then, to convert this into a bit stream (since linearization tables are bit streams written without regard to byte boundaries), you can pipe the resulting data through the following perl code:

© Copyright 2005-2021 Jay Berkenbilt, 2022-2026 Jay Berkenbilt and Manfred Holger.

**Examples:**

Example 1 (c):
```c
use bytes;
binmode STDIN;
undef $/;
my $a = <STDIN>;
my @ch = split(//, $a);
map { printf("%08b", ord($_)) } @ch;
print "\n";
```

Example 2 (c):
```c
use bytes;
binmode STDIN;
undef $/;
my $a = <STDIN>;
my @ch = split(//, $a);
map { printf("%08b", ord($_)) } @ch;
print "\n";
```

---

## Object and Cross-Reference Streams

**URL:** https://qpdf.readthedocs.io/en/stable/object-streams.html

**Contents:**
- Object and Cross-Reference Streams
- Object Streams
- Cross-Reference Streams
  - Cross-Reference Stream Data
- Implications for Linearized Files
- Implementation Notes

This chapter provides information about the implementation of object stream and cross-reference stream support in qpdf.

Object streams can contain any regular object except the following:

objects with generation > 0

the encryption dictionary

objects containing the /Length of another stream

In addition, Adobe reader (at least as of version 8.0.0) appears to not be able to handle having the document catalog appear in an object stream if the file is encrypted, though this is not specifically disallowed by the specification.

There are additional restrictions for linearized files. See Implications for Linearized Files for details.

The PDF specification refers to objects in object streams as “compressed objects” regardless of whether the object stream is compressed.

The generation number of every object in an object stream must be zero. It is possible to delete and replace an object in an object stream with a regular object.

The object stream dictionary has the following keys:

/N: number of objects

/First: byte offset of first object

/Extends: indirect reference to stream that this extends

Stream collections are formed with /Extends. They must form a directed acyclic graph. These can be used for semantic information and are not meaningful to the PDF document’s syntactic structure. Although qpdf preserves stream collections, it never generates them and doesn’t make use of this information in any way.

The specification recommends limiting the number of objects in object stream for efficiency in reading and decoding. Acrobat 6 uses no more than 100 objects per object stream for linearized files and no more 200 objects per stream for non-linearized files. QPDFWriter, in object stream generation mode, never puts more than 100 objects in an object stream.

Object stream contents consists of N pairs of integers, each of which is the object number and the byte offset of the object relative to the first object in the stream, followed by the objects themselves, concatenated.

For non-hybrid files, the value following startxref is the byte offset to the xref stream rather than the word xref.

For hybrid files (files containing both xref tables and cross-reference streams), the xref table’s trailer dictionary contains the key /XRefStm whose value is the byte offset to a cross-reference stream that supplements the xref table. A PDF 1.5-compliant application should read the xref table first. Then it should replace any object that it has already seen with any defined in the xref stream. Then it should follow any /Prev pointer in the original xref table’s trailer dictionary. The specification is not clear about what should be done, if anything, with a /Prev pointer in the xref stream referenced by an xref table. The QPDF class ignores it, which is probably reasonable since, if this case were to appear for any sensible PDF file, the previous xref table would probably have a corresponding /XRefStm pointer of its own. For example, if a hybrid file were appended, the appended section would have its own xref table and /XRefStm. The appended xref table would point to the previous xref table which would point the /XRefStm, meaning that the new /XRefStm doesn’t have to point to it.

Since xref streams must be read very early, they may not be encrypted, and the may not contain indirect objects for keys required to read them, which are these:

/Size: value n+1: where n is highest object number (same as /Size in the trailer dictionary)

/Index (optional): value [:samp:`{n count}` ...] used to determine which objects’ information is stored in this stream. The default is [0 /Size].

/Prev: value offset: byte offset of previous xref stream (same as /Prev in the trailer dictionary)

/W [...]: sizes of each field in the xref table

The other fields in the xref stream, which may be indirect if desired, are the union of those from the xref table’s trailer dictionary.

The stream data is binary and encoded in big-endian byte order. Entries are concatenated, and each entry has a length equal to the total of the entries in /W above. Each entry consists of one or more fields, the first of which is the type of the field. The number of bytes for each field is given by /W above. A 0 in /W indicates that the field is omitted and has the default value. The default value for the field type is 1. All other default values are 0.

PDF 1.5 has three field types:

0: for free objects. Format: 0 obj next-generation, same as the free table in a traditional cross-reference table

1: regular non-compressed object. Format: 1 offset generation

2: for objects in object streams. Format: 2 object-stream-number index, the number of object stream containing the object and the index within the object stream of the object.

It seems standard to have the first entry in the table be 0 0 0 instead of 0 0 ffff if there are no deleted objects.

For linearized files, the linearization dictionary, document catalog, and page objects may not be contained in object streams.

Objects stored within object streams are given the highest range of object numbers within the main and first-page cross-reference sections.

It is okay to use cross-reference streams in place of regular xref tables. There are on special considerations.

Hint data refers to object streams themselves, not the objects in the streams. Shared object references should also be made to the object streams. There are no reference in any hint tables to the object numbers of compressed objects (objects within object streams).

When numbering objects, all shared objects within both the first and second halves of the linearized files must be numbered consecutively after all normal uncompressed objects in that half.

There are three modes for writing object streams: disable, preserve, and generate. In disable mode, we do not generate any object streams, and we also generate an xref table rather than xref streams. This can be used to generate PDF files that are viewable with older readers. In preserve mode, we write object streams such that written object streams contain the same objects and /Extends relationships as in the original file. This is equal to disable if the file has no object streams. In generate, we create object streams ourselves by grouping objects that are allowed in object streams together in sets of no more than 100 objects. We also ensure that the PDF version is at least 1.5 in generate mode, but we preserve the version header in the other modes. The default is preserve.

We do not support creation of hybrid files. When we write files, even in preserve mode, we will lose any xref tables and merge any appended sections.

© Copyright 2005-2021 Jay Berkenbilt, 2022-2026 Jay Berkenbilt and Manfred Holger.

---

## QDF Mode

**URL:** https://qpdf.readthedocs.io/en/stable/qdf.html

**Contents:**
- QDF Mode

In QDF mode, qpdf creates PDF files in what we call QDF form. A PDF file in QDF form, sometimes called a QDF file, is a completely valid PDF file that has %QDF-1.0 as its third line (after the pdf header and binary characters) and has certain other characteristics. The purpose of QDF form is to make it possible to edit PDF files, with some restrictions, in an ordinary text editor. This can be very useful for experimenting with different PDF constructs or for making one-off edits to PDF files (though there are other reasons why this may not always work). Note that QDF mode does not support linearized files. If you enable linearization, QDF mode is automatically disabled.

It is ordinarily very difficult to edit PDF files in a text editor for two reasons: most meaningful data in PDF files is compressed, and PDF files are full of offset and length information that makes it hard to add or remove data. A QDF file is organized in a manner such that, if edits are kept within certain constraints, the fix-qdf program, distributed with qpdf, is able to restore edited files to a correct state.

With no arguments, fix-qdf reads the possibly-edited QDF file from standard input and writes a repaired file to standard output. You can also specify the input and output files as command-line arguments. With one argument, the argument is taken as an input file. With two arguments, the first argument is an input file, and the second is an output file.

For another way to work with PDF files in an editor, see qpdf JSON. Using qpdf JSON format allows you to edit the PDF file semantically without having to be concerned about PDF syntax. However, QDF files are actually valid PDF files, so the feedback cycle may be faster if previewing with a PDF reader. Also, since QDF files are valid PDF, you can experiment with all aspects of the PDF file, including syntax.

The following attributes characterize a QDF file:

All objects appear in numerical order in the PDF file, including when objects appear in object streams.

Objects are printed in an easy-to-read format, and all line endings are normalized to UNIX line endings.

Unless specifically overridden, streams appear uncompressed (when qpdf supports the filters and they are compressed with a non-lossy compression scheme), and most content streams are normalized (line endings are converted to just a UNIX-style linefeeds).

All streams lengths are represented as indirect objects, and the stream length object is always the next object after the stream. If the stream data does not end with a newline, an extra newline is inserted, and a special comment appears after the stream indicating that this has been done.

If the PDF file contains object streams, if object stream n contains k objects, those objects are numbered from n+1 through n+k, and the object number/offset pairs appear on a separate line for each object. Additionally, each object in the object stream is preceded by a comment indicating its object number and index. This makes it very easy to find objects in object streams.

All beginnings of objects, stream tokens, endstream tokens, and endobj tokens appear on lines by themselves. A blank line follows every endobj token.

If there is a cross-reference stream, it is unfiltered.

Page dictionaries and page content streams are marked with special comments that make them easy to find.

Comments precede each object indicating the object number of the corresponding object in the original file.

When editing a QDF file, any edits can be made as long as the above constraints are maintained. This means that you can freely edit a page’s content without worrying about messing up the QDF file. It is also possible to add new objects so long as those objects are added after the last object in the file or subsequent objects are renumbered. If a QDF file has object streams in it, you can always add the new objects before the xref stream and then change the number of the xref stream, since nothing generally ever references it by number.

It is not generally practical to remove objects from QDF files without messing up object numbering, but if you remove all references to an object without removing the object itself (by removing all indirect objects that point to it), this will leave the object unreferenced. Then you can run qpdf on the file (after running fix-qdf), and qpdf will omit the now-orphaned object.

When fix-qdf is run, it goes through the file and recomputes the following parts of the file:

the /N, /W, and /First keys of all object stream dictionaries

the pairs of numbers representing object numbers and offsets of objects in object streams

the cross-reference table or cross-reference stream

the offset to the cross-reference table or cross-reference stream following the startxref token

© Copyright 2005-2021 Jay Berkenbilt, 2022-2026 Jay Berkenbilt and Manfred Holger.

**Examples:**

Example 1 (unknown):
```unknown
fix-qdf [infilename [outfilename]]
```

Example 2 (unknown):
```unknown
fix-qdf [infilename [outfilename]]
```

---

## qpdf JSON

**URL:** https://qpdf.readthedocs.io/en/stable/json.html

**Contents:**
- qpdf JSON
- Overview
- JSON Terminology
- What qpdf JSON is not
- qpdf JSON Format
  - qpdf JSON Object Representation
  - qpdf JSON Example
  - qpdf JSON Input
  - qpdf JSON Workflow: CLI
  - qpdf JSON Workflow: API

Beginning with qpdf version 11.0.0, the qpdf library and command-line program can produce a JSON representation of a PDF file. qpdf version 11 introduces JSON format version 2. Prior to qpdf 11, versions 8.3.0 onward had a more limited JSON representation accessible only from the command-line. For details on what changed, see Changes from JSON v1 to v2. The rest of this chapter documents qpdf JSON version 2.

Please note: this chapter discusses qpdf JSON format, which represents the contents of a PDF file. This is distinct from the QPDFJob JSON format which provides a higher-level interface interacting with qpdf the way the command-line tool does. For information about that, see QPDFJob: a Job-Based Interface.

The qpdf JSON format is specific to qpdf. The --json command-line flag causes creation of a JSON representation the objects in a PDF file along with JSON-formatted summaries of other information about the file. This functionality is built into QPDFJob and can be accessed from the qpdf command-line tool or from the QPDFJob C or C++ API.

Starting with qpdf JSON version 2, from qpdf 11.0.0, the JSON output includes an unambiguous and complete representation of the PDF objects and header. The information without the JSON-formatted summaries of other information is also available using the QPDF::writeJSON method.

By default, stream data is omitted from the JSON data, but it can be included by specifying the --json-stream-data option. With stream data included, the generated JSON file completely represents a PDF file. You can think of this as using JSON as an alternative syntax for representing a PDF file. Using qpdf JSON, it is possible to convert a PDF file to JSON, manipulate the structure or contents of the objects at a low level, and convert the results back to a PDF file. This functionality can be accessed from the command-line with the --json-input, and --update-from-json flags, or from the API using the QPDF::createFromJSON, and QPDF::updateFromJSON methods. The --json-output flag changes a handful of defaults so that the resulting JSON is as close as possible to the original input and is ready for being converted back to PDF.

The qpdf JSON data includes unreferenced objects. This may be addressed in a future version of qpdf. For now, that means that certain objects that are not useful in the JSON representation are included. This includes linearization and encryption dictionaries, linearization hint streams, object streams, and the cross-reference (xref) stream associated with the trailer dictionary where applicable. For the best experience with qpdf JSON, you can run the file through qpdf first to remove encryption, linearization, and object streams. For example:

Notes about terminology:

In JavaScript and JSON, that thing that has keys and values is typically called an object.

In PDF, that thing that has keys and values is typically called a dictionary. An object is a PDF object such as integer, real, boolean, null, string, array, dictionary, or stream.

Some languages that use JSON call an object a dictionary, a map, or a hash.

Sometimes, it’s called an object if it has fixed keys and a dictionary if it has variable keys.

This manual is not entirely consistent about its use of dictionary vs. object because sometimes one term or another is clearer in context. Just be aware of the ambiguity when reading the manual. We frequently use the term dictionary to refer to a JSON object because of the consistency with PDF terminology, particular when referring to a dictionary that contains information PDF objects.

Please note that qpdf JSON offers a convenient syntax for manipulating PDF files at a low level using JSON syntax. JSON syntax is much easier to work with than native PDF syntax, and there are good JSON libraries in virtually every commonly used programming language. Working with PDF objects in JSON removes the need to worry about stream lengths, cross reference tables, and PDF-specific representations of Unicode or binary strings that appear outside of content streams. It does not eliminate the need to understand the semantic structure of PDF files. Working with qpdf JSON still requires familiarity with the PDF specification.

In particular, qpdf JSON does not provide any of the following capabilities:

Text extraction. While you could use qpdf JSON syntax to navigate to a page’s content streams and font structures, text within pages is still encoded using PDF syntax within content streams, and there is no assistance for text extraction.

Reflowing text, document structure. qpdf JSON does not add any new information or insight into the content of PDF files. If you have a PDF file that lacks any structural information, qpdf JSON won’t help you solve any of those problems.

This is what we mean when we say that JSON provides a alternative syntax for working with PDF data. Semantically, it is identical to native PDF.

This section describes how qpdf represents PDF objects in JSON format. It also describes how to work with qpdf JSON to create or modify PDF files.

This section describes the representation of PDF objects in qpdf JSON version 2. An example appears in qpdf JSON Example.

PDF objects are represented within the "qpdf" entry of a qpdf JSON file. The "qpdf" entry is a two-element array. The first element is a dictionary containing header-like information about the file such as the PDF version. The second element is a dictionary containing all the objects in the PDF file. We refer to this as the objects dictionary.

The first element contains the following keys:

"jsonversion" – a number indicating the JSON version used for writing. This will always be 2.

"pdfversion" – a string containing PDF version as indicated in the PDF header (e.g. "1.7", "2.0")

pushedinheritedpageresources – a boolean indicating whether the library pushed inherited resources down to the page level. Certain library calls cause this to happen, and qpdf needs to know when reading a JSON file back in whether it should do this as it may cause certain objects to be renumbered. This field is ignored when --update-from-json was not given.

calledgetallpages – a boolean indicating whether getAllPages was called prior to writing the JSON output. This method causes page tree repair to occur, which may renumber some objects (in very rare cases of corrupted page trees), so qpdf needs to know this information when reading a JSON file back in. This field is ignored when --update-from-json was not given.

"maxobjectid" – a number indicating the object ID of the highest numbered object in the file. This is provided to make it easier for software that wants to add new objects to the file as you can safely start with one above that number when creating new objects. Note that the value of "maxobjectid" may be higher than the actual maximum object that appears in the input PDF since it takes into consideration any dangling indirect object references from the original file. This prevents you from unwittingly creating an object that doesn’t exist but that is referenced, which may have unintended side effects. (The PDF specification explicitly allows dangling references and says to treat them as nulls. This can happen if objects are removed from a PDF file.)

The second element is the objects dictionary. Each key in the objects dictionary is either "trailer" or a string of the form "obj:O G R" where O and G are the object and generation numbers and R is the literal string R. This is the PDF syntax for the indirect object reference prepended by obj:. The value, representing the object itself, is a JSON object whose structure is described below.

Stream objects are represented as a JSON object with the single key "stream". The stream object has a key called "dict" whose value is the stream dictionary as an object value (described below) with the "/Length" key omitted. Other keys are determined by the value for json stream data (--json-stream-data, or a parameter of type qpdf_json_stream_data_e) as follows:

none: stream data is not represented; no other keys are present specified.

inline: the stream data appears as a base64-encoded string as the value of the "data" key

file: the stream data is written to a file, and the path to the file is stored in the "datafile" key. A relative path is interpreted as relative to the current directory when qpdf is invoked.

Keys other than "dict", "data", and "datafile" are ignored. This is primarily for future compatibility in case a newer version of qpdf includes additional information.

As with the native PDF representation, the stream data must be consistent with whatever filters and decode parameters are specified in the stream dictionary.

Non-stream objects are represented as a dictionary with the single key "value". Other keys are ignored for future compatibility. The value’s structure is described in “Object Values” below.

Note: in files that use object streams, the trailer “dictionary” is actually a stream, but in the JSON representation, the value of the "trailer" key is always written as a dictionary (with a "value" key like other non-stream objects). There will also be a a stream object whose key is the object ID of the cross-reference stream, even though this stream will generally be unreferenced. This makes it possible to assume "trailer" points to a dictionary without having to consider whether the file uses object streams or not. It is also consistent with how QPDF::getTrailer behaves in the C++ API.

Within "value" or "stream"."dict", PDF objects are represented as follows:

Objects of type Boolean or null are represented as JSON objects of the same type.

Objects that are numeric are represented as numeric in the JSON without regard to precision. Internally, qpdf stores numeric values as strings, so qpdf will preserve arbitrary precision numerical values when reading and writing JSON. It is likely that other JSON readers and writers will have implementation-dependent ways of handling numerical values that are out of range.

Name objects are represented as JSON strings that start with / and are followed by the PDF name in canonical form with all PDF syntax resolved. For example, the name whose canonical form (per the PDF specification) is text/plain would be represented in JSON as "/text/plain" and in PDF as "/text#2fplain". Starting with qpdf 11.7.0, the syntax "n:/pdf-syntax" is accepted as an alternative. This can be used for any name (e.g. "n:/text#2fplain"), but it is necessary when the name contains binary characters. For example, /one#a0two must be represented as "n:/one#a0two" since the single byte a0 is not valid in JSON.

Indirect object references are represented as JSON strings that look like a PDF indirect object reference and have the form "O G R" where O and G are the object and generation numbers and R is the literal string R. For example, "3 0 R" would represent a reference to the object with object ID 3 and generation 0.

PDF strings are represented as JSON strings in one of two ways:

"u:utf8-encoded-string": this format is used when the PDF string can be unambiguously represented as a Unicode string and contains no unprintable characters. This is the case whether the input string is encoded as UTF-16, UTF-8 (as allowed by PDF 2.0), or PDF doc encoding. Strings are only represented this way if they can be encoded without loss of information.

"b:hex-string": this format is used to represent any binary string value that can’t be represented as a Unicode string. hex-string must have an even number of characters that range from a through f, A through F, or 0 through 9.

qpdf writes empty strings as "u:", but both "b:" and "u:" are valid representations of the empty string.

There is full support for UTF-16 surrogate pairs. Binary strings encoded with "b:..." are the internal PDF representations. As such, the following are equivalent:

"u:\ud83e\udd54" – representation of U+1F954 as a surrogate pair in JSON syntax

"b:FEFFD83EDD54" – representation of U+1F954 as the bytes of a UTF-16 string in PDF syntax with the leading FEFF indicating UTF-16

"b:efbbbff09fa594" – representation of U+1F954 as the bytes of a UTF-8 string in PDF syntax (as allowed by PDF 2.0) with the leading EF, BB, BF sequence (which is just UTF-8 encoding of FEFF).

A JSON string whose contents are u: followed by the UTF-8 representation of U+1F954. This is the potato emoji. Unfortunately, I am not able to render it in the PDF version of this manual.

PDF arrays are represented as JSON arrays of objects as described above

PDF dictionaries are represented as JSON objects whose keys are the string representations of names and whose values are representations of PDF objects.

Note that writing JSON output is done by QPDF, not QPDFWriter. As such, none of the things QPDFWriter does apply. This includes recompression of streams, renumbering of objects, removal of unreferenced objects, encryption, decryption, linearization, QDF mode, etc. See Writing PDF Files for a more in-depth discussion. This has a few noteworthy implications:

Decryption is handled transparently by qpdf. As there are no qpdf APIs, even internal to the library, that allow retrieval of encrypted data in its raw, encrypted form, qpdf JSON always includes decrypted data. It is possible that a future version of qpdf may allow access to raw, encrypted string and stream data.

Objects that are related to a PDF file’s structure, rather than its content, are included in the JSON output, even though they are not particularly useful. In a future version of qpdf, this may be fixed, and the --preserve-unreferenced flag may be able to be used to get the existing behavior. For now, to avoid this, run the file through qpdf --decrypt --object-streams=disable in.pdf out.pdf to generate a new PDF file that contains no unreferenced or structural objects.

Linearized PDF files include a linearization dictionary which is not referenced from any other object and which references the linearization hint stream by offset. The JSON from a linearized PDF file contains both of these objects, even though they are not useful in the JSON. Offset information is not represented in the JSON, so there’s no way to find the linearization hint stream from the JSON. If a new PDF is created from JSON that was written, the objects will be read back in but will just be unreferenced objects that will be ignored by QPDFWriter when the file is rewritten.

The JSON from a file with object streams will include the original object stream and will also include all the objects in the stream as top-level objects.

In files with object streams, the trailer “dictionary” is a stream. In qpdf JSON files, the "trailer" key will contain a dictionary with all the keys in it relating to the stream, and the stream will also appear as an unreferenced object.

Encrypted files are decrypted, but the encryption dictionary still appears in the JSON output.

The JSON below shows an example of a simple PDF file represented in qpdf JSON format.

The qpdf JSON output can be used in two different ways:

By using the --json-input flag or calling QPDF::createFromJSON in place of QPDF::processFile, a qpdf JSON file can be used in place of a PDF file as the input to qpdf.

By using the --update-from-json flag or calling QPDF::updateFromJSON on an initialized QPDF object, a qpdf JSON file can be used to apply changes to an existing QPDF object. That QPDF object can have come from any source including a PDF file, a qpdf JSON file, or the result of any other process that results in a valid, initialized QPDF object.

Here are some important things to know about qpdf JSON input.

When a qpdf JSON file is used as the primary input file, it must be complete. This means

A JSON version number must be specified with the "jsonversion" key in the first array element

A PDF version number must be specified with the "pdfversion" key in the first array element

Stream data must be present for all streams

The trailer dictionary must be present, though only the "/Root" key is required.

Certain fields from the input are ignored whether creating or updating from a JSON file:

"maxobjectid" is ignored, so it is not necessary to update it when adding new objects.

"/Length" is ignored in all stream dictionaries. qpdf doesn’t put it there when it creates JSON output, and it is not necessary to add it.

"/Size" is ignored if it appears in a trailer dictionary as that is always recomputed by QPDFWriter.

Unknown keys at the top level of the file, within "qpdf", and at the top level of each individual PDF object (inside the dictionary that has the "value" or "stream" key) and directly within "stream" are ignored for future compatibility. This includes other top-level keys generated by qpdf itself (such as "pages"). As such, those keys don’t have to be consistent with the "qpdf" key if modifying a JSON file for conversion back to PDF. If you wish to store application-specific metadata, you can do so by adding a key whose name starts with x-. qpdf is guaranteed not to add any of its own keys that starts with x-. Note that any "version" key at the top level is ignored. The JSON version is obtained from the "jsonversion" key of the first element of the "qpdf" field.

The values of "calledgetallpages" and "pushedinheritedpageresources" are ignored when creating a file. When updating a file, they are treated as false if omitted.

When qpdf reads a PDF file, the internal object numbers are always preserved. However, when qpdf writes a file using QPDFWriter, QPDFWriter does its own numbering and, in general, does not preserve input object numbers. That means that a qpdf JSON file that is used to update an existing PDF must have object numbers that match the input file it is modifying. In practical terms, this means that you can’t use a JSON file created from one PDF file to modify the output of running qpdf on that file.

To put this more concretely, the following is valid:

By contrast, the following will produce unpredictable and probably incorrect results because out.pdf won’t have the same object numbers as pdf.json and in.pdf.

When updating from a JSON file (--update-from-json, QPDF::updateFromJSON), existing objects are updated in place. This has the following implications:

If the object you are updating is a stream, you may omit both "data" and "datafile". In that case the original stream data is preserved. You must always provide a stream dictionary, but it may be empty. Note that an empty stream dictionary will clear the old dictionary. There is no way to indicate that an old stream dictionary should be left alone, so if your intention is to replace the stream data and preserve the dictionary, the original dictionary must appear in the JSON file.

You can change one object type to another object type including replacing a stream with a non-stream or a non-stream with a stream. If you replace a non-stream with a stream, you must provide data for the stream.

Objects that you do not wish to modify can be omitted from the JSON. That includes the trailer. That means you can use the output of a qpdf JSON file that was written using --json-object to have it include only the objects you intend to modify.

You can omit the "pdfversion" key. The input PDF version will be preserved.

This section includes a few examples of using qpdf JSON.

Convert a PDF file to JSON format, edit the JSON, and convert back to PDF. This is an alternative to using QDF mode (see QDF Mode) to modify PDF files in a text editor. Each method has its own advantages and disadvantages.

Extract only a specific object into a JSON file, modify the object in JSON, and use the modified object to update the original PDF. In this case, we’re editing object 4, whatever that may happen to be. You would have to know through some other means which object you wanted to edit, such as by looking at other JSON output or using a tool (possibly but not necessarily qpdf) to identify the object.

Rather than using --json-object as in the above example, you could edit the JSON file to remove the objects you didn’t need. You could also just leave them there, though the update process would be slower.

You could also add new objects to a file by adding them to pdf.json. Just be sure the object number doesn’t conflict with an existing object. The "maxobjectid" field in the original output can help with this. You don’t have to update it if you add objects as it is ignored when the file is read back in.

Use --json-input and --json-output together to demonstrate preservation of object numbers. In this example, a.json and b.json will have the same objects and object numbers. The files may not be identical since strings may be normalized, fields may appear in a different order, etc. However b.json and c.json are probably identical.

Everything that can be done using the qpdf CLI can be done using the C++ API. See comments in QPDF.hh for writeJSON, createFromJSON, and updateFromJSON for details.

The qpdf JSON representation includes a JSON serialization of the raw objects in the PDF file as well as some computed information in a more easily extracted format. qpdf provides some guarantees about its JSON format. These guarantees are designed to simplify the experience of a developer working with the JSON format.

The top-level JSON object is a dictionary (JSON “object”). The JSON output contains various nested dictionaries and arrays. With the exception of dictionaries that are populated by the fields of PDF objects from the file, all instances of a dictionary are guaranteed to have exactly the same keys.

The top-level JSON structure contains a "version" key whose value is simple integer. The value of the version key will be incremented if a non-compatible change is made. A non-compatible change would be any change that involves removal of a key, a change to the format of data pointed to by a key, or a semantic change that requires a different interpretation of a previously existing key. Note that, starting with version 2, the JSON version also appears in the "jsonversion" field of the first element of "qpdf" field.

Within a specific qpdf JSON version, future versions of qpdf are free to add additional keys but not to remove keys or change the type of object that a key points to. That means that consumers of qpdf JSON should ignore keys they don’t know about.

The qpdf command can be invoked with the --json-help option. This will output a JSON structure that has the same structure as the JSON output that qpdf generates, except that each field in the help output is a description of the corresponding field in the JSON output. The specific guarantees are as follows:

A dictionary in the help output means that the corresponding location in the actual JSON output is also a dictionary with exactly the same keys; that is, no keys present in help are absent in the real output, and no keys will be present in the real output that are not in help. It is possible for a key to be present and have a value that is explicitly null. As a special case, if the dictionary has a single key whose name starts with < and ends with >, it means that the JSON output is a dictionary that can have any value as a key. This is used for cases in which the keys of the dictionary are things like object IDs.

A string in the help output is a description of the item that appears in the corresponding location of the actual output. The corresponding output can have any value including null.

A single-element array in the help output indicates that the corresponding location in the actual output is either a single item or is an array of any length. The single item or each element of the array has whatever format is implied by the single element of the help output’s array.

A multi-element array in the help output indicates that the corresponding location in the actual output is an array of the same length. Each element of the output array has whatever format is implied by the corresponding element of the help output’s array.

For example, the help output indicates includes a "pagelabels" key whose value is an array of one element. That element is a dictionary with keys "index" and "label". In addition to describing the meaning of those keys, this tells you that the actual JSON output will contain a pagelabels array, each of whose elements is a dictionary that contains an index key, a label key, and no other keys.

The JSON output contains the value of every object in the file, but it also contains some summary data. This is analogous to how qpdf’s library interface works. The summary data is similar to the helper functions in that it allows you to look at certain aspects of the PDF file without having to understand all the nuances of the PDF specification, while the raw objects allow you to mine the PDF for anything that the higher-level interfaces are lacking. It is especially useful to create a JSON file with the "pages" and "qpdf" keys and to use the "pages" information to find a page rather than navigating the pages tree manually. This can be done safely, and changes can made to the objects dictionary without worrying about keeping "pages" up to date since it is ignored when reading the file back in.

For the most part, the built-in JSON help tells you everything you need to know about the JSON format, but there are a few non-obvious things to be aware of:

If a PDF file has certain types of errors in its pages tree (such as page objects that are direct or multiple pages sharing the same object ID), qpdf will automatically repair the pages tree. If you specify "qpdf" (or, with qpdf JSON version 1, "objects" or "objectinfo") without any other keys, you will see the original pages tree without any corrections. If you specify any of keys that require page tree traversal (for example, "pages", "outlines", or "pagelabel"), then "qpdf" (and "objects" and "objectinfo") will show the repaired page tree so that object references will be consistent throughout the file. You can tell if this has happened by looking at the "calledgetallpages" and "pushedinheritedpageresources" fields in the first element of the "qpdf" array.

While qpdf guarantees that keys present in the help will be present in the output, those fields may be null or empty if the information is not known or absent in the file. Also, if you specify --json-key, the keys that are not listed will be excluded entirely except for those that --json-help says are always present.

In a few places, there are keys with names containing pageposfrom1. The values of these keys are null or an integer. If an integer, they point to a page index within the file numbering from 1. Note that JSON indexes from 0, and you would also use 0-based indexing using the API. However, 1-based indexing is easier in this case because the command-line syntax for specifying page ranges is 1-based. If you were going to write a program that looked through the JSON for information about specific pages and then use the command-line to extract those pages, 1-based indexing is easier. Besides, it’s more convenient to subtract 1 in a real programming language than it is to add 1 in shell code.

The image information included in the page section of the JSON output includes the key "filterable". Note that the value of this field may depend on the --decode-level that you invoke qpdf with. The JSON output includes a top-level key "parameters" that indicates the decode level that was used for computing whether a stream was filterable. For example, jpeg images will be shown as not filterable by default, but they will be shown as filterable if you run qpdf --json --decode-level=all.

The encrypt key’s values will be populated for non-encrypted files. Some values will be null, and others will have values that apply to unencrypted files.

The qpdf library itself never loads an entire PDF into memory. This remains true for PDF files represented in JSON format. In general, qpdf will hold the entire object structure in memory once a file has been fully read (objects are loaded into memory lazily but stay there once loaded), but it will never have more than two copies of a stream in memory at once. That said, if you ask qpdf to write JSON to memory, it will do so, so be careful about this if you are working with very large PDF files. There is nothing in the qpdf library itself that prevents working with PDF files much larger than available system memory. qpdf can both read and write such files in JSON format. If you need to work with a PDF file’s json representation in memory, it is recommended that you use either none or file as the argument to --json-stream-data, or if using the API, use qpdf_sj_none or pdf_sj_file as the json stream data value. If using none, you can use other means to obtain the stream data.

The following changes were made to qpdf’s JSON output format for version 2.

The representation of objects has changed. For details, see qpdf JSON Object Representation.

The representation of strings is now unambiguous for all strings. Strings a prefixed with either u: for Unicode strings or b: for byte strings.

Names are shown in qpdf’s canonical form rather than in PDF syntax. (Example: the PDF-syntax name /text#2fplain appeared as "/text#2fplain" in v1 but appears as "/text/plain" in v2. In qpdf 11.7.0, a fix was made to accept "n:/pdf-syntax" for names containing binary characters.

The top-level representation of an object in "objects" is a dictionary containing either a "value" key or a "stream" key, making it possible to distinguish streams from other objects.

The "objectinfo" and "objects" keys have been removed in favor of a representation in "qpdf" that includes header information and differentiates between a stream and other kinds of objects. In v1, it was not possible to tell a stream from a dictionary within "objects", and the PDF version was not captured at all.

Within the objects dictionary, keys are now "obj:O G R" where O and G are the object and generation number. "trailer" remains the key for the trailer dictionary. In v1, the obj: prefix was not present. The rationale for this change is as follows:

Having a unique prefix (obj:) makes it much easier to search in the JSON file for the definition of an object

Having the key still contain O G R makes it much easier to construct the key from an indirect reference. You just have to prepend obj:. There is no need to parse the indirect object reference.

In the "encrypt" object, the "modifyannotations" was misspelled as "moddifyannotations" in v1. This has been corrected.

qpdf JSON version 2 was created to make it possible to manipulate PDF files using JSON syntax instead of native PDF syntax. This makes it possible to make low-level updates to PDF files from just about any programming language or even to do so from the command-line using tools like jq or any editor that’s capable of working with JSON files. There were several limitations of JSON format version 1 that made this impossible:

Strings, names, and indirect object references in the original PDF file were all converted to strings in the JSON representation. For casual human inspection, this was fine, but in the general case, there was no way to tell the difference between a string that looked like a name or indirect object reference from an actual name or indirect object reference.

PDF strings were not unambiguously represented in the JSON format. The way qpdf JSON v1 represented a string was to try to convert the string to UTF-8. This was done by assuming a string that was not explicitly marked as Unicode was encoded in PDF doc encoding. The problem is that there is not a perfect bidirectional mapping between Unicode and PDF doc encoding, so if a binary string happened to contain characters that couldn’t be bidirectionally mapped, there would be no way to get back to the original PDF string. Even when possible, trying to map from the JSON representation of a binary string back to the original string required knowledge of the mapping between PDF doc encoding and Unicode.

There was no representation of stream data. If you wanted to extract stream data, you could use --show-object, so this wasn’t that important for inspection, but it was a blocker for being able to go from JSON back to PDF. qpdf JSON version 2 allows stream data to be included inline as base64-encoded data. There is also an option to write all stream data to external files, which makes it possible to work with very large PDF files in JSON format even with tools that try to read the entire JSON structure into memory.

The PDF version from PDF header was not represented in qpdf JSON v1.

© Copyright 2005-2021 Jay Berkenbilt, 2022-2026 Jay Berkenbilt and Manfred Holger.

**Examples:**

Example 1 (unknown):
```unknown
qpdf --decrypt --object-streams=disable in.pdf out.pdf
qpdf --json-output out.pdf out.json
```

Example 2 (unknown):
```unknown
qpdf --decrypt --object-streams=disable in.pdf out.pdf
qpdf --json-output out.pdf out.json
```

Example 3 (json):
```json
{
  "qpdf": [
    {
      "jsonversion": 2,
      "pdfversion": "1.3",
      "pushedinheritedpageresources": false,
      "calledgetallpages": false,
      "maxobjectid": 6
    },
    {
      "obj:1 0 R": {
        "value": {
          "/Pages": "3 0 R",
          "/Type": "/Catalog"
        }
      },
      "obj:2 0 R": {
        "value": {
          "/Author": "u:Digits of π",
          "/CreationDate": "u:D:20220731155308-05'00'",
          "/Creator": "u:A person typing in Emacs",
          "/Keywords": "u:potato, example",
          "/ModDate": "u:D:20220731155308-05'00'",
          "/Producer": "u:qpdf",
          "/Subject": "u:Example",
          "/Title": "u:Something potato-related"
        }
      },
      "obj:3 0 R": {
        "value": {
          "/Count": 1,
          "/Kids": [
            "4 0 R"
          ],
          "/Type": "/Pages"
        }
      },
      "obj:4 0 R": {
        "value": {
          "/Contents": "5 0 R",
          "/MediaBox": [
            0,
            0,
            612,
            792
          ],
          "/Parent": "3 0 R",
          "/Resources": {
            "/Font": {
              "/F1": "6 0 R"
            }
          },
          "/Type": "/Page"
        }
      },
      "obj:5 0 R": {
        "stream": {
          "data": "eJxzCuFSUNB3M1QwMlEISQOyzY2AyEAhJAXI1gjIL0ksyddUCMnicg3hAgDLAQnI",
          "dict": {
            "/Filter": "/FlateDecode"
          }
        }
      },
      "obj:6 0 R": {
        "value": {
          "/BaseFont": "/Helvetica",
          "/Encoding": "/WinAnsiEncoding",
          "/Subtype": "/Type1",
          "/Type": "/Font"
        }
      },
      "trailer": {
        "value": {
          "/ID": [
            "b:98b5a26966fba4d3a769b715b2558da6",
            "b:6bea23330e0b9ff0ddb47b6757fb002e"
          ],
          "/Info": "2 0 R",
          "/Root": "1 0 R",
          "/Size": 7
        }
      }
    }
  ]
}
```

Example 4 (json):
```json
{
  "qpdf": [
    {
      "jsonversion": 2,
      "pdfversion": "1.3",
      "pushedinheritedpageresources": false,
      "calledgetallpages": false,
      "maxobjectid": 6
    },
    {
      "obj:1 0 R": {
        "value": {
          "/Pages": "3 0 R",
          "/Type": "/Catalog"
        }
      },
      "obj:2 0 R": {
        "value": {
          "/Author": "u:Digits of π",
          "/CreationDate": "u:D:20220731155308-05'00'",
          "/Creator": "u:A person typing in Emacs",
          "/Keywords": "u:potato, example",
          "/ModDate": "u:D:20220731155308-05'00'",
          "/Producer": "u:qpdf",
          "/Subject": "u:Example",
          "/Title": "u:Something potato-related"
        }
      },
      "obj:3 0 R": {
        "value": {
          "/Count": 1,
          "/Kids": [
            "4 0 R"
          ],
          "/Type": "/Pages"
        }
      },
      "obj:4 0 R": {
        "value": {
          "/Contents": "5 0 R",
          "/MediaBox": [
            0,
            0,
            612,
            792
          ],
          "/Parent": "3 0 R",
          "/Resources": {
            "/Font": {
              "/F1": "6 0 R"
            }
          },
          "/Type": "/Page"
        }
      },
      "obj:5 0 R": {
        "stream": {
          "data": "eJxzCuFSUNB3M1QwMlEISQOyzY2AyEAhJAXI1gjIL0ksyddUCMnicg3hAgDLAQnI",
          "dict": {
            "/Filter": "/FlateDecode"
          }
        }
      },
      "obj:6 0 R": {
        "value": {
          "/BaseFont": "/Helvetica",
          "/Encoding": "/WinAnsiEncoding",
          "/Subtype": "/Type1",
          "/Type": "/Font"
        }
      },
      "trailer": {
        "value": {
          "/ID": [
            "b:98b5a26966fba4d3a769b715b2558da6",
            "b:6bea23330e0b9ff0ddb47b6757fb002e"
          ],
          "/Info": "2 0 R",
          "/Root": "1 0 R",
          "/Size": 7
        }
      }
    }
  ]
}
```

---
