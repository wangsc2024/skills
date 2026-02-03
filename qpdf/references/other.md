# Qpdf - Other

**Pages:** 5

---

## Contributing to qpdf

**URL:** https://qpdf.readthedocs.io/en/stable/contributing.html

**Contents:**
- Contributing to qpdf
- Source Repository
- Code Formatting
- Automated Tests
  - Coverage
- Personal Comments

The qpdf source code lives at https://github.com/qpdf/qpdf.

Create issues (bug reports, feature requests) at https://github.com/qpdf/qpdf/issues. If you have a general question or topic for discussion, you can create a discussion at https://github.com/qpdf/qpdf/discussions.

The qpdf source code is formatted using clang-format with a .clang-format file at the top of the source tree. The format-code script reformats all the source code in the repository. You must have clang-format in your path, and it must be at least version 20.

For emacs users, the .dir-locals.el file configures emacs cc-mode for an indentation style that is similar to but not exactly like what clang-format produces. When there are differences, clang-format is authoritative. It is not possible to make cc-mode and clang-format exactly match since the syntax parser in emacs is not as sophisticated.

Blocks of code that should not be formatted can be surrounded by the comments // clang-format off and // clang-format on. Sometimes clang-format tries to combine lines in ways that are undesirable. In this case, we follow a convention of adding a comment // line-break on its own line.

For exact details, consult .clang-format. Here is a broad, partial summary of the formatting rules:

Use spaces, not tabs.

Keep lines to 100 columns when possible.

Braces are on their own lines after classes and functions (and similar top-level constructs) and are compact otherwise.

Closing parentheses are attached to the previous material, not not their own lines.

The README-maintainer file has a few additional notes that are probably not important to anyone who is not making deep changes to qpdf.

The testing style of qpdf has evolved over time. More recent tests call assert(). Older tests print stuff to standard output and compare the output against reference files. Many tests are a mixture of these techniques.

The qtest style of testing is to test everything through the application. So effectively most testing is “integration testing” or “end-to-end testing”.

For details about qtest, consult the QTest Manual. As you read it, keep in mind that, in spite of the recent date on the file, the vast majority of that documentation is from before 2007 and predates many test frameworks and approaches that are in use today.

In most cases, things in the code are tested through integration tests, though the test suite is very thorough. Many tests are driven through the qpdf CLI. Others are driven through other files in the qpdf directory, especially test_driver.cc and qpdf-ctest.c. These programs only use the public API.

In some cases, there are true “unit tests”, but they are exercised through various stand-alone programs that exercise the library in particular ways, including some that have access to library internals. These are in the libtests directory.

You wil see calls to QTC::TC throughout the code. This is a “manual coverage” system described in depth in the qtest documentation linked above. It works by ensuring that QTC::TC is called sometime during the test in each configured way. In brief:

QTC::TC takes two mandatory options and an optional one:

The first two arguments must be string literals. This is because qtest finds coverage cases lexically.

The first argument is the scope name, usually qpdf. This means there is a qpdf.testcov file in the source directory.

The second argument is a case name. Each case name appears in qpdf.testcov with a number after it, usually 0.

If the third argument is present, it is a number. qtest ensures that the QTC::TC is called for that scope and case at least once with the third argument set to every value from 0 to n inclusive, where n is the number after the coverage call.

QTC::TC does nothing unless certain environment variables are set. Therefore, QTC:TC calls should have no side effects. (In some languages, they may be disabled at compile-time, though qpdf does not actually do this.)

So, for example, if you have this code:

and this line in qpdf.testcov:

the test suite will only pass if that line of code was called at least once with skipped_space == 0 and at least once with skipped_space == 1.

The manual coverage approach ensures the reader that certain conditions were covered in testing. Use of QTC::TC is only part of the overall strategy.

I do not require testing on pull requests, but they are appreciated, and I will not merge any code that is not tested. Often someone will submit a pull request that is not adequately tested but is a good contribution. In those cases, I will often take the code, add it with tests, and accept the changes that way rather than merging the pull request as submitted.

qpdf started as a work project in 2002. The first open source release was in 2008. While there have been a handful of contributors, the vast majority of the code was written by one person over many years as a side project.

I maintain a very strong commitment to backward compatibility. As such, there are many aspects of the code that are showing their age. While I believe the codebase to have high quality, there are things that I would do differently if I were doing them from scratch today. Sometimes people will suggest changes that I like but can’t accept for backward compatibility reasons.

While I welcome contributions and am eager to collaborate with contributors, I have a high bar. I only accept things I’m willing to maintain over the long haul, and I am happy to help people get submissions into that state.

© Copyright 2005-2021 Jay Berkenbilt, 2022-2026 Jay Berkenbilt and Manfred Holger.

**Examples:**

Example 1 (yaml):
```yaml
QTC::TC("qpdf", "QPDF eof skipping spaces before xref",
        skipped_space ? 0 : 1);
```

Example 2 (yaml):
```yaml
QTC::TC("qpdf", "QPDF eof skipping spaces before xref",
        skipped_space ? 0 : 1);
```

Example 3 (unknown):
```unknown
QPDF eof skipping spaces before xref 1
```

Example 4 (unknown):
```unknown
QPDF eof skipping spaces before xref 1
```

---

## Notes for Packagers

**URL:** https://qpdf.readthedocs.io/en/stable/packaging.html

**Contents:**
- Notes for Packagers
- Build Options
- Package Tests
- Packaging Documentation
  - Documentation Packaging Rationale

If you are packaging qpdf for an operating system distribution, this chapter is for you. Otherwise, feel free to skip.

For a detailed discussion of build options, please refer to Build Options. This section calls attention to options that are particularly useful to packagers.

Perl must be present at build time. Prior to qpdf version 9.1.1, there was a runtime dependency on perl, but this is no longer the case.

Make sure you are getting the intended behavior with regard to crypto providers. Read Build-time Crypto Selection for details.

Use of SHOW_FAILED_TEST_OUTPUT is recommended for building in continuous integration or other automated environments as it makes it possible to see test failures in build logs. This should be combined with either ctest --verbose or ctest --output-on-failure.

Packagers should never define the FUTURE build option. API changes enabled by FUTURE are not stable and may be API/ABI-breaking. That option is intended only for people who are testing their code with a local build of qpdf to provide early feedback or prepare for possible future changes to the API.

qpdf’s install targets do not install completion files by default since there is no standard location for them. As a packager, it’s good if you install them wherever your distribution expects such files to go. You can find completion files to install in the completions directory. See the completions/README.md file for more information.

Starting with qpdf 11, qpdf’s default installation installs source files from the examples directory with documentation. Prior to qpdf 11, this was a recommendation for packagers but was not done automatically.

Starting with qpdf 11.10, qpdf can be built with zopfli support (see Building with zopfli support). It is recommended not to build qpdf with zopfli for distributions since it adds zopfli as a dependency, and this library is less widely used that qpdf’s other dependencies. Users who want that probably know they want it, and they can compile from source. Note that, per zopfli’s own documentation, zopfli is about 100 times slower than zlib and produces compression output about 5% smaller.

The pkg-test directory contains very small test shell scripts that are designed to help smoke-test an installation of qpdf. They were designed to be used with debian’s autopkgtest framework but can be used by others. Please see pkg-test/README.md in the source distribution for details.

Starting in qpdf version 10.5, pre-built documentation is no longer distributed with the qpdf source distribution. Here are a few options you may want to consider for your packages:

When you run make install, the file README-doc.txt is installed in the documentation directory. That file tells the reader where to find the documentation online and where to go to download offline copies of the documentation. This is the option selected by the debian packages.

Embed pre-built documentation

You can obtain pre-built documentation and extract its contents into your distribution. This is what the Windows binary distributions available from the qpdf release site do. You can find the pre-built documentation in the release area in the file qpdf-version-doc.zip. For an example of this approach, look at qpdf’s GitHub actions build scripts. The build-scripts/build-doc script builds with -DBUILD_DOC_DIST=1 to create the documentation distribution. The build-scripts/build-windows script extracts it into the build tree and builds with -DINSTALL_MANUAL=1 to include it in the installer.

Build the documentation yourself

You can build the documentation as part of your build process. Be sure to pass -DBUILD_DOC_DIST=1 and -DINSTALL_MANUAL=1 to cmake. This is what the AppImage build does. The latest version of Sphinx at the time of the initial conversion a sphinx-based documentation was 4.3.2. Older versions are not guaranteed to work.

This section describes the reason for things being the way they are. It’s for information only; you don’t have to know any of this to package qpdf.

What is the reason for this change? Prior to qpdf 10.5, the qpdf manual was a docbook XML file. The generated documents were the product of running the file through build-time style sheets and contained no copyrighted material of their own. Starting with version 10.5, the manual is written in reStructured Text and built with Sphinx. This change was made to make it much easier to automatically generate portions of the documentation and to make the documentation easier to work with. The HTML output of Sphinx is also much more readable, usable, and suitable for online consumption than the output of the docbook style sheets. The downsides are that the generated HTML documentation now contains Javascript code and embedded fonts, and the PDF version of the documentation is no longer as suitable for printing (at least as of the 10.5 distribution) since external link targets are no longer shown and cross references no longer contain page number information. The presence of copyrighted material in the generated documentation, even though things are licensed with MIT and BSD licenses, complicates the job of the packager in various ways. For one thing, it means the NOTICE.md file in the source repository would have to keep up with the copyright information for files that are not controlled in the repository. Additionally, some distributions (notably Debian/Ubuntu) discourage inclusion of sphinx-generated documentation in packages, preferring you instead to build the documentation as part of the package build process and to depend at runtime on a shared package that contains the code. At the time of the conversion of the qpdf manual from docbook to sphinx, newer versions of both sphinx and the html theme were required than were available in some of most of the Debian/Ubuntu versions for which qpdf was packaged.

Since always-on Internet connectivity is much more common than it used to be, many users of qpdf would prefer to consume the documentation online anyway, and the lack of pre-built documentation in the distribution won’t be as big of a deal. However there are still some people who can’t or choose not to view documentation online. For them, pre-built documentation is still available.

© Copyright 2005-2021 Jay Berkenbilt, 2022-2026 Jay Berkenbilt and Manfred Holger.

---

## qpdf version 12.3.0

**URL:** https://qpdf.readthedocs.io/en/stable/index.html

**Contents:**
- qpdf version 12.3.0
- Indices

Welcome to the qpdf documentation! For the latest version of this documentation, please visit https://qpdf.readthedocs.io.

qpdf Command-line Options

© Copyright 2005-2021 Jay Berkenbilt, 2022-2026 Jay Berkenbilt and Manfred Holger.

---

## qpdf version 12.3.0

**URL:** https://qpdf.readthedocs.io/en/stable/

**Contents:**
- qpdf version 12.3.0
- Indices

Welcome to the qpdf documentation! For the latest version of this documentation, please visit https://qpdf.readthedocs.io.

qpdf Command-line Options

© Copyright 2005-2021 Jay Berkenbilt, 2022-2026 Jay Berkenbilt and Manfred Holger.

---

## Weak Cryptography

**URL:** https://qpdf.readthedocs.io/en/stable/weak-crypto.html

**Contents:**
- Weak Cryptography
- Definition of Weak Cryptographic Algorithm
- Uses of Weak Encryption in qpdf
- Uses of Weak Hashing In qpdf
- API-Breaking Changes in qpdf 11.0

For help with compiler errors in qpdf 11.0 or newer, see API-Breaking Changes in qpdf 11.0.

Since 2006, the PDF specification has offered ways to create encrypted PDF files without using weak cryptography, though it took a few years for many PDF readers and writers to catch up. It is still necessary to support weak encryption algorithms to read encrypted PDF files that were created using weak encryption algorithms, including all PDF files created before the modern formats were introduced or widely supported.

Starting with version 10.4, qpdf began taking steps to reduce the likelihood of a user accidentally creating PDF files with insecure cryptography but will continue to allow creation of such files indefinitely with explicit acknowledgment. The restrictions on use of weak cryptography were made stricter with qpdf 11.

We divide weak cryptographic algorithms into two categories: weak encryption and weak hashing. Encryption is encoding data such that a key of some sort is required to decode it. Hashing is creating a short value from data in such a way that it is extremely improbable to find two documents with the same hash (known has a hash collision) and extremely difficult to intentionally create a document with a specific hash or two documents with the same hash.

When we say that an encryption algorithm is weak, we either mean that a mathematical flaw has been discovered that makes it inherently insecure or that it is sufficiently simple that modern computer technology makes it possible to use “brute force” to crack. For example, when 40-bit keys were originally introduced, it wasn’t practical to consider trying all possible keys, but today such a thing is possible.

When we say that a hashing algorithm is weak, we mean that, either because of mathematical flaw or insufficient complexity, it is computationally feasible to intentionally construct a hash collision.

While weak encryption should always be avoided, there are cases in which it is safe to use a weak hashing algorithm when security is not a factor. For example, a weak hashing algorithm should not be used as the only mechanism to test whether a file has been tampered with. In other words, you can’t use a weak hash as a digital signature. There is no harm, however, in using a weak hash as a way to sort or index documents as long as hash collisions are tolerated. It is also common to use weak hashes as checksums, which are often used a check that a file wasn’t damaged in transit or storage, though for true integrity, a strong hash would be better.

Note that qpdf must always retain support for weak cryptographic algorithms since this is required for reading older PDF files that use it. Additionally, qpdf will always retain the ability to create files using weak cryptographic algorithms since, as a development tool, qpdf explicitly supports creating older or deprecated types of PDF files since these are sometimes needed to test or work with older versions of software. Even if other cryptography libraries drop support for RC4 or MD5, qpdf can always fall back to its internal implementations of those algorithms, so they are not going to disappear from qpdf.

When PDF files are encrypted using 40-bit encryption or 128-bit encryption without AES, then the weak RC4 algorithm is used. You can avoid using weak encryption in qpdf by always using 256-bit encryption. Unless you are trying to create files that need to be opened with PDF readers from before about 2010 (by which time most readers had added support for the stronger encryption algorithms) or are creating insecure files explicitly for testing or some similar purpose, there is no reason to use anything other than 256-bit encryption.

By default, qpdf refuses to write a file that uses weak encryption. You can explicitly allow this by specifying the --allow-weak-crypto option.

In qpdf 11, all library methods that could potentially cause files to be written with weak encryption were deprecated, and methods to enable weak encryption were either given explicit names indicating this or take required arguments to enable the insecure behavior.

There is one exception: when encryption parameters are copied from the input file or another file to the output file, there is no prohibition or even warning against using insecure encryption. The reason is that many qpdf operations simply preserve whatever encryption is there, and requiring confirmation to preserve insecure encryption would cause qpdf to break when non-encryption-related operations were performed on files that happened to be encrypted. Failing or generating warnings in this case would likely have the effect of making people use the --allow-weak-crypto option blindly, which would be worse than just letting those files go so that explicit, conscious selection of weak crypto would be more likely to be noticed. Why, you might ask, does this apply to --copy-encryption as well as to the default behavior preserving encryption? The answer is that --copy-encryption works with an unencrypted file as input, which enables workflows where one may start with a file, decrypt it just in case, perform a series of operations, and then reapply the original encryption, if any. Also, one may have a template used for encryption that one may apply to a variety of output files, and it would be annoying to be warned about it for every output file.

The PDF specification makes use the weak MD5 hashing algorithm in several places. While it is used in the encryption algorithms, breaking MD5 would not be adequate to crack an encrypted file when 256-bit encryption is in use, so using 256-bit encryption is adequate for avoiding the use of MD5 for anything security-sensitive.

MD5 is used in the following non-security-sensitive ways:

Generation of the document ID. The document ID is an input parameter to the document encryption but is not itself considered to be secure. They are supposed to be unique, but they are not tamper-resistant in non-encrypted PDF files, and hash collisions must be tolerated.

The PDF specification recommends but does not require the use of MD5 in generation of document IDs. Usually there is also a random component to document ID generation. There is a qpdf-specific feature of generating a deterministic ID (see --deterministic-id) which also uses MD5. While it would certainly be possible to change the deterministic ID algorithm to not use MD5, doing so would break all previous deterministic IDs (which would render the feature useless for many cases) and would offer very little benefit since even a securely generated document ID is not itself a security-sensitive value.

Checksums in embedded file streams – the PDF specification specifies the use of MD5.

It is therefore not possible completely avoid the use of MD5 with qpdf, but as long as you are using 256-bit encryption, it is not used in a security-sensitive fashion.

In qpdf 11, several deprecated functions and methods were removed. These methods provided an incomplete API. Alternatives were added in qpdf 8.4.0. The removed functions are

C API: qpdf_set_r3_encryption_parameters, qpdf_set_r4_encryption_parameters, qpdf_set_r5_encryption_parameters, qpdf_set_r6_encryption_parameters

QPDFWriter: overloaded versions of these methods with fewer arguments: setR3EncryptionParameters, setR4EncryptionParameters, setR5EncryptionParameters, and setR6EncryptionParameters

Additionally, remaining functions/methods had their names changed to signal that they are insecure and to force developers to make a decision. If you intentionally want to continue to use insecure cryptographic algorithms and create insecure files, you can change your code just add _insecure or Insecure to the end of the function as needed. (Note the disappearance of 2 in some of the C functions as well.) Better, you should migrate your code to use more secure encryption as documented in QPDFWriter.hh. Use the R6 methods (or their corresponding C functions) to create files with 256-bit encryption.

qpdf_set_r2_encryption_parameters

qpdf_set_r2_encryption_parameters_insecure

qpdf_set_r3_encryption_parameters2

qpdf_set_r3_encryption_parameters_insecure

qpdf_set_r4_encryption_parameters2

qpdf_set_r2_encryption_parameters_insecure

QPDFWriter::setR2EncryptionParameters

QPDFWriter::setR2EncryptionParametersInsecure

QPDFWriter::setR3EncryptionParameters

QPDFWriter::setR3EncryptionParametersInsecure

QPDFWriter::setR4EncryptionParameters

QPDFWriter::setR4EncryptionParametersInsecure

© Copyright 2005-2021 Jay Berkenbilt, 2022-2026 Jay Berkenbilt and Manfred Holger.

---
