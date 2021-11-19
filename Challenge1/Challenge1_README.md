# HACKtheMACHINE 2021 | Track 2: Data Science Detective Bot



## Challenge 1: Build a Classifier

For the first challenge, you will be exploring the dataset provided to develop the best possible prediction model, applying ML/AI to malware classification. The goal is to build a binary classifier that can take PE file data (represented as `EMBER` features) as input and outputs a malware/benign decision. This README will provide more details on the input data, how to submit and how you’ll be graded.

`EMBER`: https://github.com/elastic/ember



## Starter Notebooks

 A Starter Notebook can be found in the Challenge 1 repository. This notebook provides particpants with a headstart on the challenge 1. In the notebook below you will find code to: load libraries, read in files, and initial exploratory data analysis.


* `Starter_Notebook_1_EMBER_Data.ipynb` - This notebook takes a look at our datasets and provides some resources to help get you started.


## Performance Metrics

Performance metrics are used to evaluate your machine learning algorithm and pipeline. Metrics provide insight into how your model is performing under the hood.

| Metric | Definition | Formula |
|--------|------------|---------|
|Precision| Fraction of relevant instances among the retrieved instances| Precision = TP / (TP + FP) |
|Recall (Sensitivity) | Fraction of relevant instances that were retrieved | Recall = TP / (TP + FN) |
|F1-Score | Measure of a tests accuracy, the harmonic mean of the precision and recall | F1-Score = (2 x Precision x Recall ) / (Precision + Recall ) |


Sources: https://en.wikipedia.org/wiki/Precision_and_recall, https://en.wikipedia.org/wiki/F-score

## Data Folder

In `/Data` you'll find two training datasets `flatten_train.xlsx` and `raw_train.xlsx`. Both datasets have the same data (rows) with different features.

There is no test dataset because your solution will be tested when you submit your Docker image.

## Data Dictionary

The following data dictionary describes the columns or fields of the data set and a description of the objects. This information with more detail can also be found in the `EMBER` documentation of the `features.py` file at: https://github.com/elastic/ember/blob/master/ember/features.py


| Field Name | Description |
|------------|-------------|
| sha256 | The Secure Hash Algorithm (SHA) is a cryptographic hash function like a signature or fingerprints for a data set. Even if one symbol is changed the algorithm will produce a different hash value. The SHA256 algorithm generates a fixed size 256-bit (32-byte) hash. The SHA256 algorithm is used to ensure you acquire the same data as the original. For exmaple, if you download something you can check if the data has not changed (due to network errors or malware injection) by comparing the hashes of your file and the original.|
| histogram | Byte histogram (count + non-normalized) over the entire binary file. The byte histogram contains 256 integer values and represent the counts of each byte value within the value. When generating model features the byte histgoram is normalized to a distribution, since file size is represented as a feature in the general file information. |
| byteentropy | 2D byte/entropy histogram based loosely on (Saxe and Berlin, 2015). This roughly approximates the joint probability of byte value and local entropy. See Section 2.1.1 in https://arxiv.org/pdf/1508.03096.pdf for more info. The byte entropy histogram approximates the joint distriubtion p(H, X) of entropy H and byte value X. By computing the scalar entropy H for a fixed-length window and pairing it with each byte occurrence within the window. This is repeated as the window slides across the input bytes. |
| strings | Contains simple statistics about printable strings of the following: <ul><li>`numstrings`: number of strings <li> `avlength`: average length of strings <li>`printabledist`: histogram of the printable characters within those strings <li>`printables`: distinct information from byte histogram information from the byte histogram information since its derived only from strings containing at least 5 consecutive printable characters <li>`entropy`: entropy of characters across all printable strings <li>`paths` number of strings that begin with **C:** (case insensitive) that may indicate a path <li>`urls`: the number of occurences of **http://** or **https://** (case insensitive) that may indicate a URL <li>`registry`: number of occurrences of HKEY that may indicate a registry key, <li>`MZ`: number of occurrences of the short string MZ |
| general | Provides general file information. 0/1 indicates a binary output <ul><li>`size`: length of bytes <li>`vsize`: virtual size <li>`has_debug`: 0/1  <li>`exports`: 0/1 <li>`imports`: 0/1 <li>`has_relocations`: 0/1  <li>`has_resources`: 0/1 <li>`has_signature`: 0/1 <li>`has_tls`: 0/1 <li>`symbols`: 0/1 |
| header | Provides header file information on machine, architecture, OS, link and other information: <ul><li> `coeff`: [ `timestamp`, `machine`,`characteristics` ] <li> `optional`: [`subsystem`, `dll_characteristics`, `magic`, `major_image_version`, `minor_linker_version`, `major_operating_system_version`, `minor_operating_system_version`, `major_subsystem_version`, `minor_subsystem_version`, `sizeof_code`, `sizeof_headers`, `sizeof_heap_commit`]
| section | Information about section names, sizes and entropy. Uses hashing trick to summarize all this section into a feature vector. <ul><li> `imports`: [`KERNEL32.dll` : [`GetTickCount`] |
| imports | Information about imported libraries and functions from the import address table. Note that the total number of imported functions is contained in GeneralFileInfo. |
| exports | Information about exported functions. Note that the total number of exported functions is contained in GeneralFileInfo.|
| datadirectories | Extracts size and virtual address of the first 15 data dictectories. |
| label / category | Class label indicating benign `0` or malicious `1`|

## Recommended Resources

### Python Tutorials:
* [Python Fundamentals](https://www.tutorialspoint.com/python/index.htm)


### Python Libraries:
* [Pandas](https://pandas.pydata.org/docs/) - Data analysis and manipulation
* [Matplotlib](https://matplotlib.org/stable/contents.html) - Data visualizations and plots
* [EMBER - Elastic Malware Benchmark for Empowering Researchers](https://github.com/elastic/ember)
*[Sklearn](https://scikit-learn.org/stable/index.html) - Machine Learning in Python

### Papers/Articles:
* [EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models](https://arxiv.org/pdf/1804.04637.pdf) - Original Paper for the EMBER library
* [The rise of machine learning for detection and classification of malware: Research developments, trends, and challenges](https://reader.elsevier.com/reader/sd/pii/S1084804519303868?token=1B85475195D5551845E9E32F74B919C68B83EA9FA742D8101974DAB0C602DE47AF793D75C2ED033A60E68D54DB1F7998&originRegion=us-east-1&originCreation=20211112194210)
* [A brief introduction to PE format](https://medium.com/ax1al/a-brief-introduction-to-pe-format-6052914cc8dd)
* Refer to `PE Files Explained.docx` document provided

## Challenge 1 Submission Instructions

Challenge 1 submission has two parts:
1) Any code used to train a classifier or perform analysis should be checked into the appropriate folder in your designated GitLab repo (i.e. Challenge 1 code in the `Challenge 1` folder).
2) A docker image that contains your model and inference code will be pushed to your designated image registry.

Please reference the files already in the Submission folder for an example

## Challenge 1 Submission Deadline

Please submit your work by 1400 EST on Wednesday, 17 November.

## Challenge 1 Grading

Model performance will be tested using a data set similar to the ones given as training data.

**Challenge 1 will be worth a total of 25 points.**

| Criteria | Description | Criteria % Weight |
|----------|-------------|-------------------|
|**Model Performance** |Model F1 Score | 95 |
|**Documentation** | Any code used to train models, perform analysis in git with reasonable levels of commenting. |5 |
