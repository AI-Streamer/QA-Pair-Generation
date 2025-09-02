# QA Pair Generation

Utilize LLM to generate the QA pairs from plain texts for style tranferring.

\[EN|[ZH](README-ZH.md)\]

## Usage

**Please adjust VarMap.py file in the corresponding directory before generating QA pairs.**

| File              | Functionality                                                   |
| ----------------- | --------------------------------------------------------------- |
| PairGeneration.py | Script for generate QA pairs                                    |
| System-Prompt.txt | System prompt                                                   |
| Human-Prompt.txt  | Human prompt                                                    |
| VarMap.py         | Organize the directory of plain text file and QA pair JSON file |

Run `PairGeneration.py` to generate the QA pairs. Although this respository aims to generate QA pairs that imitate the language style in the plain text, you can also apply the script in generating knowlegde QA pairs for RAG by changing the system prompt.

Currently, the prompts are in simplified Chinese since the plain text is in Chinese and the model applied is Qwen3-Max.

## Catalog

- [ ] Google Colab inference example.
- [ ] Add options for API platform and English prompt.
- [ ] Post the generate QA pairs.
