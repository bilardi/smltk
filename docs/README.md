# Document development

## Monolingual

### Installation

```sh
git clone https://github.com/bilardi/smltk
cd smltk/docs
pip3 install --upgrade -r requirements.txt # install your dependences
sphinx-quickstart
```

### Usage

For testing the web changes,

```sh
cd smltk/docs
make html
# go to your browser at file://path/of/the/folder/of/smltk/docs/build/html/index.html
```

For testing the pdf changes, check if there are some lines into [conf.py](https://github.com/bilardi/smltk/blob/master/docs/source/conf.py)

```sh
extensions = ['rst2pdf.pdfbuilder']
pdf_documents = [('index', u'rst2pdf', u'Sample rst2pdf doc', u'Your Name'),]
  # index - master document
  # rst2pdf - name of the generated pdf
  # Sample rst2pdf doc - title of the pdf
  # Your Name - author name in the pdf
```

and then type

```sh
cd smltk/docs
sphinx-build -b pdf source build/pdf
open build/pdf/smltk.pdf
```
