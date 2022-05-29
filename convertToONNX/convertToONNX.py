import io
import os

import torch
import torch.onnx
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from transformers import convert_graph_to_onnx as onnx_converter

tokenizer = AutoTokenizer.from_pretrained("csarron/bert-base-uncased-squad-v1")
model = AutoModelForQuestionAnswering.from_pretrained("csarron/bert-base-uncased-squad-v1")

from pathlib import Path

def Convert_ONNX():  

    qa_pipeline = pipeline(
        "question-answering",
        model,
        tokenizer = tokenizer
    )
    predictions = qa_pipeline({
        'context': "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain \"Amazonas\" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species.",
        'question': "How many square kilometers of rainforest is covered in the basin?"
    })

    print(predictions)

    model.eval()
    onnx_converter.convert(framework="pt",
                           model=model,
                           output=Path("onnx/bert-base-uncased-squad.onnx"),
                           opset=11,
                           tokenizer=tokenizer,
                           user_external_format=False,
                           pipeline_name="question-answering")

    print(" ")
    print('Model has been converted to ONNX')

if __name__ == "__main__":
    Convert_ONNX()



