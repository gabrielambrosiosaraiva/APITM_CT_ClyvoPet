# CLYVO VET — Apoio à teleconsulta

Aplicação de demonstração que classifica a raça de um pet a partir de uma imagem, consulta informações relacionadas no Oracle e sugere perguntas para apoiar a conversa entre veterinário e tutor.

> A classificação e as perguntas são recursos de apoio à teleconsulta. Elas não substituem a avaliação nem o diagnóstico do médico-veterinário.

## Demonstração

Aplicação publicada: [celticstechclyvovet.streamlit.app](https://celticstechclyvovet.streamlit.app/)

## Funcionalidades

- Recebe imagens nos formatos JPG, JPEG e PNG.
- Classifica a raça usando um modelo TensorFlow Lite exportado pelo Teachable Machine.
- Exibe a raça prevista e a confiança do modelo.
- Consulta no Oracle os cuidados associados à raça quando existe um cadastro correspondente.
- Se não encontrar um cadastro específico, busca os cuidados gerais para cães (`Canis`).
- Sugere perguntas para ajudar o veterinário a conduzir a anamnese.

## Tecnologias

- Python
- Streamlit
- TensorFlow Lite
- Oracle Database, acessado com `oracledb`
- Pillow e NumPy para leitura e preparação das imagens

## Estrutura do projeto

```text
.
├── main.py
├── requirements.txt
└── modelo/
    ├── model.tflite
    └── labels.txt
```

O arquivo `main.py` espera encontrar o modelo em `modelo/model.tflite` e os rótulos em `modelo/labels.txt`.

## Executar localmente

1. Instale uma versão do Python compatível com as dependências do `requirements.txt`.
2. Na pasta principal do projeto, instale as dependências:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Configure as credenciais do Oracle, conforme a seção abaixo.
4. Inicie a aplicação:

   ```bash
   streamlit run main.py
   ```

5. Abra no navegador o endereço local mostrado no terminal.

## Configurar o Oracle

### Streamlit Community Cloud

Em **Manage app → Settings → Secrets**, configure as variáveis abaixo com os dados fornecidos para o banco Oracle:

```toml
ORACLE_USER = "seu_usuario"
ORACLE_PASSWORD = "sua_senha"
ORACLE_DSN = "oracle.fiap.com.br:1521/ORCL"
```

### Execução local

Crie um arquivo `.env` na pasta principal do projeto e preencha:

```dotenv
ORACLE_USER=seu_usuario
ORACLE_PASSWORD=sua_senha
ORACLE_DSN=oracle.fiap.com.br:1521/ORCL
```

Não envie o `.env` ao GitHub nem publique as credenciais. No Streamlit Community Cloud, use a área **Secrets**.

## Usar a aplicação

1. Abra a aplicação.
2. Envie uma foto do pet.
3. Clique em **Analisar imagem**.
4. Confira a raça prevista, os cuidados encontrados no banco e as perguntas sugeridas.

## Problemas comuns

- **Modelo não encontrado:** confira se `modelo/model.tflite` está no repositório com esse nome e caminho.
- **Arquivo de rótulos não encontrado:** confira se `modelo/labels.txt` está no repositório.
- **ORA-01017:** o Oracle recusou o usuário ou a senha; revise `ORACLE_USER` e `ORACLE_PASSWORD` nos Secrets.
- **Falha na consulta Oracle:** confira também se `ORACLE_DSN` corresponde ao endereço e serviço disponibilizados para o banco.

## Repositório

[APITM_CT_ClyvoPet no GitHub](https://github.com/gabrielambrosiosaraiva/APITM_CT_ClyvoPet)
