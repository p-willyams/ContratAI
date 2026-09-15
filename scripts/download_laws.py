import json
import re
import time
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


def baixar_html(url: str, tentativas: int = 4) -> str:
    ultimo_erro = None
    for tentativa in range(1, tentativas + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=45)
            resp.raise_for_status()
            resp.encoding = "ISO-8859-1"
            return resp.text
        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
        ) as e:
            ultimo_erro = e
            espera = 3 * tentativa
            print(
                f"  [retry {tentativa}/{tentativas}] falha ao conectar, esperando {espera}s... ({e.__class__.__name__})"
            )
            time.sleep(espera)
    raise ConnectionError(
        f"Não foi possível baixar {url} após {tentativas} tentativas. "
        f"Verifique sua conexão, VPN, firewall/antivírus, ou tente novamente mais tarde."
    ) from ultimo_erro


def extrair_texto_bruto(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    paragrafos = soup.find_all("p")
    texto = "\n".join(p.get_text(separator=" ", strip=True) for p in paragrafos)
    return texto


def limpar_texto(texto: str) -> str:
    texto = re.sub(r"\(Incluíd[oa] pela Lei[^)]*\)", "", texto)
    texto = re.sub(r"\(Reda[cç][aã]o dada pela Lei[^)]*\)", "", texto)
    texto = re.sub(r"\bVig[eê]ncia\b", "", texto)
    texto = re.sub(r"\bProdu[cç][aã]o\s+de\s+efeitos\b", "", texto)
    texto = re.sub(r"\s{2,}", " ", texto)
    return texto.strip()


CABECALHOS_ESTRUTURAIS = re.compile(
    r"^\s*(CAP[IÍ]TULO|T[IÍ]TULO|LIVRO|SE[CÇ][AÃ]O|SUBSE[CÇ][AÃ]O)\s+[IVXLCDM]+\b[^\n]*$",
    re.MULTILINE | re.IGNORECASE,
)


def remover_cabecalhos_estruturais(texto: str) -> str:
    return CABECALHOS_ESTRUTURAIS.split(texto)[0].strip()


MARCADOR_INCISO = re.compile(
    r"(?:^|(?<=[;.)\n]))\s*([IVXLCDM]{1,6})\s*-\s*", re.MULTILINE
)


def remover_incisos_revogados(texto: str) -> str:
    marcadores = list(MARCADOR_INCISO.finditer(texto))
    if len(marcadores) < 2:
        return texto
    limites = [m.start() for m in marcadores] + [len(texto)]
    segmentos = [
        (marcadores[i].group(1), texto[limites[i] : limites[i + 1]])
        for i in range(len(marcadores))
    ]
    filtrados = []
    i = 0
    while i < len(segmentos):
        num, seg = segmentos[i]
        proximo_e_duplicata_adjacente = (
            i + 1 < len(segmentos) and segmentos[i + 1][0] == num
        )
        if proximo_e_duplicata_adjacente:
            i += 1
            continue
        filtrados.append(seg)
        i += 1
    caput = texto[: marcadores[0].start()]
    return caput + "".join(filtrados)


PADRAO_CITACAO_DE_EMENDA = re.compile(r"\.{4,}")


def eh_citacao_de_emenda(corpo: str) -> bool:
    return bool(PADRAO_CITACAO_DE_EMENDA.search(corpo)) or corpo.rstrip().endswith(
        "(NR)"
    )


def dividir_por_artigo(texto: str) -> dict:
    padrao = re.compile(r"Art\.\s*(\d+)[ºo°]?\.?\s")
    partes = padrao.split(texto)
    artigos = {}
    for i in range(1, len(partes) - 1, 2):
        numero = int(partes[i])
        corpo = partes[i + 1]
        corpo = corpo.split("Art.")[0]
        if eh_citacao_de_emenda(corpo):
            continue
        if numero in artigos:
            continue
        corpo = remover_cabecalhos_estruturais(corpo)
        corpo = remover_incisos_revogados(corpo)
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
        time.sleep(3)
    saida = "../data/legislacao_clausulai.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in dataset], f, ensure_ascii=False, indent=2)
    print(f"\nConcluído: {len(dataset)} artigos salvos em {saida}")


if __name__ == "__main__":
    main()
