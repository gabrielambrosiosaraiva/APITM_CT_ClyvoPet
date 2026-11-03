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
import tensorflow as tf
from threading import Lock

PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_MODELO = PASTA_PROJETO / "modelo" / "model_unquant.tflite"
CAMINHO_ROTULOS = PASTA_PROJETO / "modelo" / "labels.txt"
TAMANHO_IMAGEM = (224, 224)

load_dotenv(PASTA_PROJETO / ".env")


def normalizar(valor: str) -> str:
    valor = unicodedata.normalize("NFKD", valor)
    valor = "".join(caractere for caractere in valor if not unicodedata.combining(caractere))
    return re.sub(r"[^a-z0-9]", "", valor.casefold())


def ler_rotulos() -> list[str]:
    rotulos: list[str] = []
    for line in CAMINHO_ROTULOS.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rotulos.append(re.sub(r"^\d+\s+", "", line))
    if not rotulos:
        raise RuntimeError("O arquivo labels.txt está vazio.")
    return rotulos


@st.cache_resource
def carregar_classificador():
    """Prepara o modelo TensorFlow Lite para classificar imagens."""
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            "Não encontrei modelo/model.tflite. Exporte o modelo em TensorFlow Lite e coloque-o nessa pasta."
        )

    interpretador = tf.lite.Interpreter(model_path=str(CAMINHO_MODELO))
    interpretador.allocate_tensors()
    return interpretador, Lock()
def conectar_oracle():
    """Connect using Streamlit Secrets online or environment variables locally."""
    def obter_configuracao(nome_variavel: str) -> str | None:
        try:
            valor = st.secrets.get(nome_variavel)
        except Exception:
            valor = None
        return valor or os.getenv(nome_variavel)

    usuario = obter_configuracao("ORACLE_USER")
    senha = obter_configuracao("ORACLE_PASSWORD")
    endereco_oracle = obter_configuracao("ORACLE_DSN")

    variaveis_faltando = [
        nome_variavel
        for nome_variavel, valor in (
            ("ORACLE_USER", usuario),
            ("ORACLE_PASSWORD", senha),
            ("ORACLE_DSN", endereco_oracle),
        )
        if not valor
    ]
    if variaveis_faltando:
        raise RuntimeError(
            "Configure ORACLE_USER, ORACLE_PASSWORD e ORACLE_DSN em Streamlit Secrets (online) ou no arquivo .env (local)."
        )

    return oracledb.connect(user=usuario, password=senha, dsn=endereco_oracle)
def classificar_imagem(imagem: Image.Image) -> tuple[str, float]:
    interpretador, cadeado = carregar_classificador()
    rotulos = ler_rotulos()
    dados_entrada = interpretador.get_input_details()[0]
    dados_saida = interpretador.get_output_details()[0]

    forma_entrada = dados_entrada["shape"]
    largura = int(forma_entrada[2])
    altura = int(forma_entrada[1])
    imagem_redimensionada = imagem.convert("RGB").resize((largura, altura))
    pixels = np.asarray(imagem_redimensionada, dtype=np.float32)
    entrada_modelo = np.expand_dims((pixels / 127.5) - 1.0, axis=0)
    entrada_modelo = entrada_modelo.astype(dados_entrada["dtype"])

    with cadeado:
        interpretador.set_tensor(dados_entrada["index"], entrada_modelo)
        interpretador.invoke()
        pontuacoes = interpretador.get_tensor(dados_saida["index"])[0]

    if len(pontuacoes) != len(rotulos):
        raise RuntimeError(
            f"O modelo retornou {len(pontuacoes)} classes, mas labels.txt contém {len(rotulos)} rótulos."
        )
    indice = int(np.argmax(pontuacoes))
    return rotulos[indice], float(pontuacoes[indice])
def buscar_cuidados_especie(raca_prevista: str) -> tuple[str, list[dict], bool]:
    """Busca cuidados da raça ou os cuidados gerais para cães."""
    conexao = conectar_oracle()
    try:
        with conexao.cursor() as cursor_oracle:
            cursor_oracle.execute(
                """
                SELECT id_especie, nome_especie
                FROM TB_CT_ESPECIE
                WHERE genero = 'Canis'
                """
            )
            especies = cursor_oracle.fetchall()

            especie_encontrada = next(
                (registro for registro in especies if normalizar(registro[1]) == normalizar(raca_prevista)),
                None,
            )
            correspondencia_exata = especie_encontrada is not None

            if correspondencia_exata:
                cursor_oracle.execute(
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
                    id_especie=especie_encontrada[0],
                )
                nome_consultado = especie_encontrada[1]
            else:
                cursor_oracle.execute(
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
                nome_consultado = "Canis (cuidados gerais para cães)"

            registros_cuidado = [
                {
                    "predisposicao": registro[0],
                    "orientacao_predisposicao": registro[1],
                    "restricao": registro[2],
                    "orientacao_restricao": registro[3],
                    "gravidade": registro[4],
                }
                for registro in cursor_oracle.fetchall()
            ]
            return nome_consultado, registros_cuidado, correspondencia_exata
    finally:
        conexao.close()


def sugerir_pergunta(condicao: str) -> str:
    termo_normalizado = normalizar(condicao)
    regras = [
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
    for palavras_chave, pergunta in regras:
        if any(normalizar(palavra_chave) in termo_normalizado for palavra_chave in palavras_chave):
            return pergunta
    return f"Você observou algum sinal relacionado a {condicao.lower()}? Quando começou e com que frequência ocorre?"


st.set_page_config(page_title="CLYVO VET | Apoio à teleconsulta", page_icon="🐾", layout="centered")
st.title("CLYVO VET — Apoio à teleconsulta")
st.write("Envie uma imagem do pet para classificar a raça, consultar os cuidados associados no Oracle e preparar perguntas de apoio ao veterinário.")

arquivo_imagem = st.file_uploader("Imagem do pet", type=["jpg", "jpeg", "png"])
if arquivo_imagem:
    imagem = Image.open(arquivo_imagem)
    st.image(imagem, caption="Imagem recebida", use_container_width=True)
    if st.button("Analisar imagem", type="primary"):
        with st.spinner("Classificando a imagem e consultando o Oracle..."):
            try:
                raca, confianca = classificar_imagem(imagem)
                st.subheader("Resultado da classificação")
                st.write(f"**Raça prevista:** {raca}")
                st.write(f"**Confiança do modelo:** {confianca:.1%}")
                nome_consultado, dados_cuidado, correspondencia_exata = buscar_cuidados_especie(raca)
                st.subheader("Informações encontradas no banco")
                st.write(f"**Cadastro consultado:** {nome_consultado}")
                if not correspondencia_exata:
                    st.info("A raça prevista não tem cadastro individual correspondente; foram consultados os cuidados gerais de Canis.")
                if not dados_cuidado:
                    st.warning("Nenhuma predisposição ou restrição foi encontrada para essa busca.")
                else:
                    perguntas = []
                    for registro_cuidado in dados_cuidado:
                        with st.expander(f"{registro_cuidado['predisposicao']} · {registro_cuidado['restricao']} ({registro_cuidado['gravidade']})"):
                            st.markdown(f"**Orientação registrada:** {registro_cuidado['orientacao_predisposicao']}")
                            st.markdown(f"**Restrição:** {registro_cuidado['orientacao_restricao']}")
                            pergunta = sugerir_pergunta(registro_cuidado["predisposicao"] + " " + registro_cuidado["restricao"])
                            st.markdown(f"**Pergunta sugerida ao tutor:** {pergunta}")
                            perguntas.append(pergunta)
                    st.subheader("Roteiro de perguntas para o veterinário")
                    for pergunta in dict.fromkeys(perguntas):
                        st.markdown(f"- {pergunta}")
                st.caption("As sugestões apoiam a anamnese; a avaliação clínica cabe ao médico-veterinário.")
            except Exception as erro:
                st.error(f"Não foi possível concluir a análise: {erro}")
