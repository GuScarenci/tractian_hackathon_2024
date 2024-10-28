import flask
import boto3
import io
import os
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os

model_name = "gpt-4o-mini"

from langchain_community.document_loaders import PyPDFLoader


ordens = [
    {
        "titulo": "trocar o óleo do motor WEG-1234",
        "ferramentas": "chave phillips tipo 1; chave torx tipo 3; óleo de motor id 90834509",
        "passos": ["remover parafuso 1, mais em cima", "tirar a tampa"],
    }
]

processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50", revision="no_timm"
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50", revision="no_timm"
)


# get the database helper functions, non-html routes and session dictionary
import db

s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT"], use_ssl=False)

# create the main flask application
app = flask.Flask("Hackaton", "")

chat = ChatOpenAI(model_name=model_name, temperature=0)

csv = open("../ferramentas.csv").read()


@app.route("/")
def root():
    return "working"


@app.route("/ordem/", methods=["POST"])
def ordempost():
    texto = flask.request.form.get("text")
    manual = flask.request.files["file"]

    manual.save("/tmp/file.pdf")

    loader = PyPDFLoader("/tmp/file.pdf")
    pages = ""
    for page in loader.load():
        pages = pages + page.page_content

    resp = chat(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "o texto abaixo contém um problema visto em uma máquina em uma fábrica. Você tem acesso à uma lista de ferramentas que podem ser usadas como um CSV, e o manual da máquina em si. Por favor, gere uma lista de ferramentas a serem usadas para resolver o problema especificado, além de uma lista de passos para resolver o problema. Não adicione nenhuma informação a mais além da lista de ferramentas e passos em uma lista não numerada. Escreva os passos da forma mais detalhada possível",
                    },
                    {
                        "type": "text",
                        "text": texto,
                    },
                    {"type": "text", "text": csv},
                    {
                        "type": "text",
                        "text": pages,
                    },
                ],
            )
        ]
    ).content
    print(resp)

    ferramentas = []
    passos = []

    get_ferramentas = True

    for line in resp.split("\n"):
        if "ferramenta" in line.lower():
            get_ferramentas = True
            continue
        if "passo" in line.lower():
            get_ferramentas = False
            continue

        if "-" not in line:
            get_ferramentas = False

        line = line.replace("-", "").strip()

        if len(line) == 0:
            continue

        if get_ferramentas:
            ferramentas.append(line)
        else:
            passos.append(line)

    print(ferramentas, passos)

    ordens.append({"titulo": texto, "ferramentas": ferramentas, "passos": passos})

    return {"id": len(ordens)-1, "titulo": texto, "ferramentas": ferramentas, "passos": passos}


@app.route("/ordem/<id>")
def ordemid(id):
    return ordens[int(id)]


@app.route("/imagem/<objeto>", methods=["POST"])
def ordemidimagem(objeto):
    file = flask.request.files["image"]
    image = Image.open(file.stream)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    d = ImageDraw.Draw(image)

    print("starting image detection")

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        obj = model.config.id2label[label.item()]
        print(
            f"Detected {obj} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
        if obj == objeto or objeto == "all":
            d.rectangle(box, outline=0xff8342, width=10)

    img_io = io.BytesIO()
    image.save(img_io, "JPEG", quality=70)
    img_io.seek(0)

    return flask.send_file(img_io, mimetype="image/jpeg")
