"""Importable ChemMiner extraction utilities.

This file preserves the public prompt/sectioning behavior from the original
Colab-oriented ChemMiner script while removing import-time side effects such as
Google Drive mounting, global OpenAI credentials, and hard-coded working
directories.  The legacy script's active extraction path was:

1. split paper text on ``".\n"``
2. find procedure anchors
3. merge nearby chunks into <=3000-character sections
4. call the coreference prompt for each section
5. call the general-procedure reaction prompt for each section

The figure-abbreviation prompt is also preserved here so benchmark wrappers can
enable that pass without mutating the vendored integration at runtime.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any


PROCEDURE_ANCHORS = (
    "General Procedure",
    "Typical Procedure",
    "General Experimental Procedure",
)

CHEMMINER_SYSTEM_MESSAGE = "You are a helpful assistant on chemistry."

COREFERENCE_EXAMPLE_USER = '''
    I am providing a paragraph from a piece of chemical literature. I would like you to help me identify instances of coreference, where a full chemical name is immediately followed by a shorthand label or alias. Here is the paragraph:

    "Tetraethyl (E)-8,9-Bis((Z)-3-ethoxy-3-oxo-2-phenylprop-1-en-1-
yl)hexadeca-1,8,15-triene-6,6,11,11-tetracarboxylate, 7c. It was
obtained from 3n (25 mg, 0.06 mmol) following the general
procedure for cycloisomerization reactions with Cp*RuCl(cod) and
purified by flash column chromatography (Hexane/AcOEt, 19:1).
Colorless oil (22 mg, 0.03 mmol, 86%);"

    Please provide the coreference in the following json format:
    {<full chemical name>: {<shorthand label or alias>}}
    Pay attention to direct aliases that come immediately after the chemical names.
    '''

COREFERENCE_EXAMPLE_ASSISTANT = '''
    {
      "Tetraethyl (E)-8,9-Bis((Z)-3-ethoxy-3-oxo-2-phenylprop-1-en-1-yl)hexadeca-1,8,15-triene-6,6,11,11-tetracarboxylate": "7c"
    }
    '''

REACTION_EXAMPLE_USER = '''Experimental Procedures and Characterization of Products.
General Procedure for the Preparation of Products.
[Ni2(iPr2Im)4(μ-COD)] (0.1 mmol, 83 mg), CsF (2 mmol, 304
mg), Ar-Bneop (2 mmol), fluoroarene, and toluene (10 mL) were
added to a Schlenk tube equipped with a magnetic stirring bar. The
reaction mixture was heated at 100 °C for 18 h, and after that H2O (5
mL) was added. The product was extracted with EtOAc (3 × 20 mL),
and then the combined organic layers were dried over Na2SO4 and
filtered, and the volatiles were removed in vacuo. The product was
purified by column chromatography on silica gel using hexane as the
eluent. The solvent of the product-containing fraction of the eluent
was evaporated in vacuo. The yields provided are based on Ar-Bneop.
Spectroscopic Data of the Products. 2,3,4,5,6-Pentafluoro-1,1′-
biphenyl (3aa). Following the general procedure, a white solid in 72%
yield (351 mg) was obtained from C6F6 (4 mmol, 462 μL) and C6H5-
Bneop (2 mmol, 380 mg). 1H NMR (500 MHz, CDCl3) δ 7.52−7.45
(m, 3 H), 7.44−7.41 (m, 2 H); 13C{1H} NMR (125 MHz, CDCl3) δ
144.2 (d of m, 1JCF = 247.3 Hz), 140.4 (d of m, 1JCF = 253.7 Hz),
137.8 (d of m, 1JCF = 250.9 Hz), 130.1 (t, 3JCF = 1.5 Hz), 129.3, 128.7,
126.4, 115.9 (m); 19F NMR (470 MHz, CDCl3) δ −143.26 (m, 2 F),
−155.65 (t, J = 21.0 Hz, 1 F), −162.27 (m, 2 F); 19F{1H} NMR (188
MHz, CDCl3) δ −143.28 (dd, J = 8.1, 22.0 Hz, 2 F), −155.68 (t, J =
21.0 Hz, 1 F), −162.31 (td, J = 8.1, 22.0 Hz, 2 F); HRMS (ASAP)
[C12H5F5] calcd 244.0306, found 244.0305.
Spectroscopic data for 3aa match with those previously reported in
the literature.3k
2,3,4,5,6-Pentafluoro-4′-methyl-1,1′-biphenyl (3ab). Following
the general procedure, a white solid in 76% yield (390 mg) was
obtained from C6F6 (4 mmol, 462 μL) and 4-CH3-C6H4-Bneop (2
mmol, 408 mg). 1H NMR (500 MHz, CDCl3) δ 7.31 (m, 4 H), 2.42
(s, 3 H); 13C{1H} NMR (125 MHz, CDCl3) δ 144.2 (d of m, 1JCF =
247.7 Hz), 140.2 (d of m, 1JCF = 253.3 Hz), 139.4, 137.8 (d of m, 1JCF
= 250.7 Hz), 130.0, 129.5, 123.4, 115.9 (m), 21.4; 19F NMR (470
MHz, CDCl3) δ −143.37 (m, 2 F), −156.15 (t, J = 18.8 Hz, 1 F),
−162.46 (m, 2 F); 19F{1H} NMR (188 MHz, CDCl3) δ −143.39 (dd,
J = 8.1, 22.8 Hz, 2 F), −156.17 (t, J = 21.0 Hz, 1 F), −162.50', could you please help me extract the information of yield/reactant/reagent/solvent/product from each reaction in the previous content in json format?
The content usually includes a general procedure, followed by the specific description of the reaction. The extraction should take into account both the general procedure, which provides the overall context, and the specific descriptions of each reaction, which offer unique details.
When a piece of information is missing from the specific description, consider the general procedure to infer the missing details. However, if there is any conflicting information, the specific description should take precedence.
    '''

REACTION_EXAMPLE_ASSISTANT = '''
    {"1": {"yield": "72%(351 mg)","reactant": "C6F6(4 mmol, 462 μL),C6H5-Bneop(2 mmol, 380 mg),fluoroarene","reagent": "[Ni2(iPr2Im)4(μ-COD)](0.1 mmol, 83 mg),CsF(2 mmol, 304 mg)","solvent": "toluene(10 mL)","product": "2,3,4,5,6-Pentafluoro-1,1ʹ-biphenyl"},
    "2": {"yield": "76%(390 mg)","reactant": "C6F6(4 mmol, 462 μL),4-CH3-C6H4-Bneop(2 mmol, 408 mg),fluoroarene ","reagent": "[Ni2(iPr2Im)4(μ-COD)](0.1 mmol, 83 mg),CsF(2 mmol, 304 mg)","solvent": "toluene(10 mL)","product": "2,3,4,5,6-Pentafluoro-4′-methyl-1,1′-biphenyl"}
    }
    '''

FIGURE_ABBREV_PROMPT = '''
Analyze the provided image and extract all the abbreviations (e.g., 1a, 2b, L1, B1, S1, etc.) and their corresponding chemical compound names in English.
Organize the extracted information into a structured JSON format. Each abbreviation should be used as a key, and its full chemical name should be the value.
Ensure that all data is accurate and properly formatted. For example:
{
  "1a": "2-Chloroquinoline",
  "B1": "Potassium carbonate",
  "S1": "EtOH/H2O (9:1)"
}
Focus on clarity, consistency, and completeness, extracting all abbreviations and their corresponding full chemical names based on the molecular structures shown in the image.
'''


def text_length(start: int, end: int, input_chunks: list[str]) -> int:
    length = 0
    for i in range(end - start + 1):
        length += len(input_chunks[start + i])
    return length


def merge(start: int, end: int, key_index: int, input_chunks: list[str]) -> tuple[list[int], int]:
    if start == key_index:
        merge_result: list[int] = []
        bar = 3000
    else:
        merge_result = [key_index]
        bar = 3000 - len(input_chunks[key_index])

    j = 0
    for j in range(end - start):
        if text_length(start, start + j, input_chunks) < bar:
            merge_result.append(start + j)
        if text_length(start, start + j, input_chunks) >= bar:
            break

    return merge_result, start + j


def build_sections(content_str: str) -> list[str]:
    """Return GPT input sections using ChemMiner's procedure-anchor logic."""
    chunks = content_str.split(".\n")
    for i in range(len(chunks)):
        chunks[i] += ".\n"

    key_index: list[int] = []
    for i, chunk in enumerate(chunks):
        if chunk.find("General Procedure") != -1:
            key_index.append(i)
        if chunk.find("Typical Procedure") != -1:
            key_index.append(i)
        if chunk.find("General Experimental Procedure") != -1:
            key_index.append(i)

    if len(key_index) == 0:
        return []

    section_list: list[list[int]] = []
    for i, key in enumerate(key_index):
        start = key
        if i < len(key_index) - 1:
            end = key_index[i + 1]
        else:
            end = key + 100 if key + 100 < len(chunks) else len(chunks)

        while start < end:
            merge_result, start = merge(start, end, key, chunks)
            section_list.append(merge_result)
            if start == end - 1:
                break
            if start >= len(chunks) - 10:
                break

    new_list: list[list[int]] = []
    for section in section_list:
        if section not in new_list:
            new_list.append(section)

    section_content: list[str] = []
    for section in new_list:
        text = ""
        for index in section:
            text += chunks[index]
        section_content.append(text)
    return section_content


def coreference_messages(content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CHEMMINER_SYSTEM_MESSAGE},
        {"role": "user", "content": COREFERENCE_EXAMPLE_USER},
        {"role": "assistant", "content": COREFERENCE_EXAMPLE_ASSISTANT},
        {
            "role": "user",
            "content": f'''I am providing a paragraph from another piece of chemical literature. Same as before, I would like you to help me identify instances of coreference, where a full chemical name is immediately followed by a shorthand label or alias.
    Here is the paragraph:  '{content}', Please provide the coreference in the same json format as before. Pay attention to direct aliases that come immediately after the chemical names.
    If there do not exist such coreference, please tell me "No coreference". Please check carefully about the full chemical name and shorthand label. The total number of coreference should be smaller than 5.
    ''',
        },
    ]


def reaction_messages(content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CHEMMINER_SYSTEM_MESSAGE},
        {"role": "user", "content": REACTION_EXAMPLE_USER},
        {"role": "assistant", "content": REACTION_EXAMPLE_ASSISTANT},
        {
            "role": "user",
            "content": f'''From the following contents: '{content}', could you please help me extract the information of yield/reactant/reagent/solvent/product from each reaction in the previous content in json format?
    The content usually includes a general procedure, followed by the specific description of the reaction. The extraction should take into account both the general procedure, which provides the overall context, and the specific descriptions of each reaction, which offer unique details.
    When a piece of information is missing from the specific description, consider the general procedure to infer the missing details. However, if there is any conflicting information, the specific description should take precedence.

    Please provide the chemical reaction details formatted as a JSON object. The structure must strictly adhere to the following requirements:
    1.The JSON object should consist exclusively of these keys: "yield", "reactant", "reagent", "solvent", and "product".
    2.If yield information is not available, the value for the "yield" key should be "No specific information about yield".
    3.The response should be clean and precise: it must not contain ellipses ("..."), backticks ("`"), or any code block identifiers such as "```json".
    Please ensure the JSON object is properly formatted with no additional characters or elements outside of the specified structure.
    ''',
        },
    ]


def figure_abbrev_text_prompt() -> str:
    return FIGURE_ABBREV_PROMPT


def encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_json_content(message: str) -> str:
    match = re.search(r"\{.*\}", message, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No JSON content found in the message!")


def parse_figure_abbrev_response(message: str) -> dict[str, str]:
    cleaned_message = extract_json_content(message)
    data = json.loads(cleaned_message)
    if not isinstance(data, dict):
        raise ValueError(f"Vision output is not a JSON object. Got: {type(data)}")
    normalized: dict[str, str] = {}
    for key, value in data.items():
        kk = str(key or "").strip()
        vv = str(value or "").strip()
        if kk:
            normalized[kk] = vv
    return normalized


def prompt_hash_payload() -> dict[str, str]:
    return {
        "coreference_example_user": COREFERENCE_EXAMPLE_USER,
        "coreference_example_assistant": COREFERENCE_EXAMPLE_ASSISTANT,
        "reaction_example_user": REACTION_EXAMPLE_USER,
        "reaction_example_assistant": REACTION_EXAMPLE_ASSISTANT,
        "figure_abbrev_prompt": FIGURE_ABBREV_PROMPT,
    }
