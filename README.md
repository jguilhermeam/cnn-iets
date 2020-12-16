# CNN-IETS

In development

This is a python implementation of CNN-IETS: a CNN-based probabilistic approach for information extraction by text segmentation.

Reference: https://doi.org/10.1145/3132847.3132962

observation 1: instead of Gennia Tagger we use NLTK Pos tagger
observation 2: instead of word2vec trained in pubmed articles we use here Google word2vec sample from NLTK

## To Execute

```
python3 main.py path/to/knowledge_base_file.xml path/to/input_file.txt
```

## To install dependencies

```
sudo apt install -y python3-pip
pip3 install -r requirements.txt
python3 utils/nltk_dependencies.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
