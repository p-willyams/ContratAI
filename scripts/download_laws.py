import json
import re
import time
import os
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup

FONTES = {
    "CC": {
        "nome": "Código Civil",
        "url": "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
    },
    "LGPD": {
        "nome": "Lei Geral de Proteção de Dados",
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm",
    },
    "LEI_9610": {
        "nome": "Lei de Direitos Autorais",
        "url": "https://www.planalto.gov.br/ccivil_03/leis/l9610.htm",
    },
}

ARTIGOS_RELEVANTES = {
    "CC": {
        594: "objeto",
        598: "prazo",
        599: "prazo",
        389: "pagamento",
        394: "pagamento",
        395: "pagamento",
        406: "pagamento",
        602: "rescisao",
        603: "rescisao",
        607: "rescisao",
        609: "rescisao",
        408: "multa",
        409: "multa",
        412: "multa",
        413: "multa",
        416: "multa",
    },
    "LGPD": {
        1: "confidencialidade",
        6: "confidencialidade",
        7: "confidencialidade",
        18: "confidencialidade",
        46: "confidencialidade",
        48: "confidencialidade",
    },
    "LEI_9610": {
        11: "propriedade_intelectual",
        17: "propriedade_intelectual",
        22: "propriedade_intelectual",
        28: "propriedade_intelectual",
        29: "propriedade_intelectual",
        49: "propriedade_intelectual",
    },
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ClausulAI-DataCollector/1.0)"}


@dataclass
class Artigo:
    id: str
    texto: str
    tipo: str
    fonte_lei: str
    referencia: str
    categoria: str
    link_original: str


def baixar_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


def extrair_texto_bruto(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    paragrafos = soup.find_all("p")
    texto = "\n".join(p.get_text(separator=" ", strip=True) for p in paragrafos)
    return texto


def limpar_texto(texto: str) -> str:
    texto = re.sub(r"\(Incluíd[oa] pela Lei[^)]*\)", "", texto)
    texto = re.sub(r"\(Reda[cç][aã]o dada pela Lei[^)]*\)", "", texto)
    texto = re.sub(r"\(Vigência\)", "", texto)
    texto = re.sub(r"\s{2,}", " ", texto)
    return texto.strip()


def dividir_por_artigo(texto: str) -> dict:
    padrao = re.compile(r"Art\.\s*(\d+)[ºo°]?\.?\s")
    partes = padrao.split(texto)
    artigos = {}
    for i in range(1, len(partes) - 1, 2):
        numero = int(partes[i])
        corpo = partes[i + 1]
        corpo = corpo.split("Art.")[0]
        artigos[numero] = limpar_texto(corpo)
    return artigos


def montar_dataset(fonte_chave: str) -> list[Artigo]:
    fonte = FONTES[fonte_chave]
    relevantes = ARTIGOS_RELEVANTES[fonte_chave]
    print(f"Baixando {fonte['nome']}...")
    html = baixar_html(fonte["url"])
    texto = extrair_texto_bruto(html)
    artigos_todos = dividir_por_artigo(texto)
    resultado = []
    for numero, categoria in relevantes.items():
        corpo = artigos_todos.get(numero)
        if not corpo:
            print(
                f"  [aviso] Art. {numero} não encontrado em {fonte_chave} - pular ou revisar regex"
            )
            continue
        resultado.append(
            Artigo(
                id=f"{fonte_chave.lower()}-art-{numero}",
                texto=f"Art. {numero}. {corpo}",
                tipo="legislacao",
                fonte_lei=fonte_chave,
                referencia=f"{fonte['nome']}, Art. {numero}",
                categoria=categoria,
                link_original=fonte["url"],
            )
        )
    print(f"  -> {len(resultado)} artigos extraídos de {len(relevantes)} esperados")
    return resultado


def main():
    dataset = []
    for fonte_chave in FONTES:
        dataset.extend(montar_dataset(fonte_chave))
        time.sleep(1)
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    os.makedirs(data_dir, exist_ok=True)
    saida = os.path.join(data_dir, "legislacao_clausulai.json")
    with open(saida, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in dataset], f, ensure_ascii=False, indent=2)
    print(f"\nConcluído: {len(dataset)} artigos salvos em {saida}")


if __name__ == "__main__":
    main()
