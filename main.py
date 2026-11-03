from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import oracledb
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "modelo" / "keras_model.h5"
LABELS_PATH = ROOT / "modelo" / "labels.txt"
IMAGE_SIZE = (224, 224)

load_dotenv(ROOT / ".env")


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def read_labels() -> list[str]:
    labels: list[str] = []
    for line in LABELS_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        labels.append(re.sub(r"^\d+\s+", "", line))
    if not labels:
        raise RuntimeError("O arquivo labels.txt está vazio.")
    return labels


@st.cache_resource
def load_classifier():
    return load_model(MODEL_PATH, compile=False)


def connect_oracle():
    """Connect using Streamlit Secrets online or environment variables locally."""
    def setting(name: str) -> str | None:
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None
        return value or os.getenv(name)

    user = setting("ORACLE_USER")
    password = setting("ORACLE_PASSWORD")
    dsn = setting("ORACLE_DSN")

    missing = [
        name
        for name, value in (
            ("ORACLE_USER", user),
            ("ORACLE_PASSWORD", password),
            ("ORACLE_DSN", dsn),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Configure ORACLE_USER, ORACLE_PASSWORD e ORACLE_DSN em Streamlit Secrets (online) ou no arquivo .env (local)."
        )

    return oracledb.connect(user=user, password=password, dsn=dsn)
def classify(image: Image.Image) -> tuple[str, float]:
    model = load_classifier()
    labels = read_labels()
    resized = image.convert("RGB").resize(IMAGE_SIZE)
    pixels = np.asarray(resized, dtype=np.float32)
    tensor = np.expand_dims((pixels / 127.5) - 1.0, axis=0)
    scores = np.asarray(model.predict(tensor, verbose=0))[0]
    if len(scores) != len(labels):
        raise RuntimeError(
            f"O modelo retornou {len(scores)} classes, mas labels.txt contém {len(labels)} rótulos."
        )
    index = int(np.argmax(scores))
    return labels[index], float(scores[index])


def fetch_species_care(predicted_breed: str) -> tuple[str, list[dict], bool]:
    """Return exact breed care when present, otherwise general Canis care."""
    connection = connect_oracle()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id_especie, nome_especie
                FROM TB_CT_ESPECIE
                WHERE genero = 'Canis'
                """
            )
            species = cursor.fetchall()

            match = next(
                (row for row in species if normalize(row[1]) == normalize(predicted_breed)),
                None,
            )
            exact_match = match is not None

            if exact_match:
                cursor.execute(
                    """
                    SELECT DISTINCT
                        p.nome_predisposicao,
                        p.orientacao_predisposicao,
                        r.nome_restricao,
                        r.orientacao_restricao,
                        r.gravidade_restricao
                    FROM TB_CT_CUIDADO c
                    JOIN TB_CT_PREDISPOSICAO_ESP p
                      ON p.id_predisposicao = c.id_predisposicao
                    JOIN TB_CT_RESTRICAO_ESP r
                      ON r.id_restricao = c.id_restricao
                    WHERE c.id_especie = :id_especie
                    ORDER BY p.nome_predisposicao, r.gravidade_restricao DESC
                    """,
                    id_especie=match[0],
                )
                selected_name = match[1]
            else:
                cursor.execute(
                    """
                    SELECT DISTINCT
                        p.nome_predisposicao,
                        p.orientacao_predisposicao,
                        r.nome_restricao,
                        r.orientacao_restricao,
                        r.gravidade_restricao
                    FROM TB_CT_ESPECIE e
                    JOIN TB_CT_CUIDADO c ON c.id_especie = e.id_especie
                    JOIN TB_CT_PREDISPOSICAO_ESP p
                      ON p.id_predisposicao = c.id_predisposicao
                    JOIN TB_CT_RESTRICAO_ESP r
                      ON r.id_restricao = c.id_restricao
                    WHERE e.genero = 'Canis'
                    ORDER BY p.nome_predisposicao, r.gravidade_restricao DESC
                    """
                )
                selected_name = "Canis (cuidados gerais para cães)"

            rows = [
                {
                    "predisposicao": row[0],
                    "orientacao_predisposicao": row[1],
                    "restricao": row[2],
                    "orientacao_restricao": row[3],
                    "gravidade": row[4],
                }
                for row in cursor.fetchall()
            ]
            return selected_name, rows, exact_match
    finally:
        connection.close()


def suggested_question(condition: str) -> str:
    key = normalize(condition)
    rules = [
        (("respiratorio", "respiracao"), "O pet apresenta tosse, cansaço fácil, respiração ruidosa ou dificuldade para respirar? Em que situações isso acontece?"),
        (("displasia", "coxofemoral", "articulacao"), "Você percebe dificuldade para levantar, mancar, evitar escadas ou sentir dor ao se movimentar?"),
        (("dermatite", "pele", "dobras", "otite"), "Notou coceira, vermelhidão, secreção, mau cheiro ou irritação na pele ou nas orelhas? Quando começou?"),
        (("obesidade", "gordura"), "Houve mudança recente no peso ou no apetite? Qual alimento e quantidade ele recebe por dia?"),
        (("ansiedade", "estresse", "comportamento"), "Como o pet reage quando fica sozinho ou quando há mudanças na rotina? Houve alteração de comportamento?"),
        (("calor", "temperatura"), "O pet demonstra desconforto no calor, procura locais mais frescos ou apresenta ofegação excessiva?"),
        (("renal", "hidratacao", "desidratacao"), "Você notou mudanças na ingestão de água, na frequência urinária ou na aparência da urina?"),
        (("cardiaco", "cardiaca"), "Ele apresenta cansaço fora do habitual, tosse, desmaios ou menor disposição para passeios?"),
        (("dentario", "dente", "oral"), "Percebe mau hálito, dificuldade para mastigar, salivação ou resistência ao toque na boca?"),
        (("digestiva", "gastrica", "alimentar", "pelos"), "Houve vômito, alteração nas fezes, desconforto após comer ou mudança no apetite?"),
    ]
    for keywords, question in rules:
        if any(normalize(keyword) in key for keyword in keywords):
            return question
    return f"Você observou algum sinal relacionado a {condition.lower()}? Quando começou e com que frequência ocorre?"


st.set_page_config(page_title="CLYVO VET | Apoio à teleconsulta", page_icon="🐾", layout="centered")
st.title("CLYVO VET — Apoio à teleconsulta")
st.write("Envie uma imagem do pet para classificar a raça, consultar os cuidados associados no Oracle e preparar perguntas de apoio ao veterinário.")

uploaded = st.file_uploader("Imagem do pet", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Imagem recebida", use_container_width=True)
    if st.button("Analisar imagem", type="primary"):
        with st.spinner("Classificando a imagem e consultando o Oracle..."):
            try:
                breed, confidence = classify(image)
                st.subheader("Resultado da classificação")
                st.write(f"**Raça prevista:** {breed}")
                st.write(f"**Confiança do modelo:** {confidence:.1%}")
                selected_name, care_rows, exact = fetch_species_care(breed)
                st.subheader("Informações encontradas no banco")
                st.write(f"**Cadastro consultado:** {selected_name}")
                if not exact:
                    st.info("A raça prevista não tem cadastro individual correspondente; foram consultados os cuidados gerais de Canis.")
                if not care_rows:
                    st.warning("Nenhuma predisposição ou restrição foi encontrada para essa busca.")
                else:
                    questions = []
                    for item in care_rows:
                        with st.expander(f"{item['predisposicao']} · {item['restricao']} ({item['gravidade']})"):
                            st.markdown(f"**Orientação registrada:** {item['orientacao_predisposicao']}")
                            st.markdown(f"**Restrição:** {item['orientacao_restricao']}")
                            question = suggested_question(item["predisposicao"] + " " + item["restricao"])
                            st.markdown(f"**Pergunta sugerida ao tutor:** {question}")
                            questions.append(question)
                    st.subheader("Roteiro de perguntas para o veterinário")
                    for question in dict.fromkeys(questions):
                        st.markdown(f"- {question}")
                st.caption("As sugestões apoiam a anamnese; a avaliação clínica cabe ao médico-veterinário.")
            except Exception as exc:
                st.error(f"Não foi possível concluir a análise: {exc}")
