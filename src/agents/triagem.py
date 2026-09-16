# %%
import dotenv

dotenv.load_dotenv()

import os
from enum import Enum
from typing import List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agno.agent import Agent
from agno.models.groq import Groq
from pydantic import BaseModel, Field

from src.tools.qdrant_search import CATEGORIAS_VALIDAS

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

Categoria = Enum("Categoria", {c.upper(): c for c in CATEGORIAS_VALIDAS})


class ItemTriagem(BaseModel):
    categoria: Categoria = Field(
        ..., description="A categoria de cláusula identificada."
    )
    presente_no_contrato: bool = Field(
        ...,
        description="""True se o contrato contém uma cláusula desta categoria, 
        False se a categoria está ausente e precisa ser sinalizada como tal.""",
    )
    trecho_do_contrato: str = Field(
        default="",
        description="""O trecho exato do contrato correspondente a esta 
        categoria. Deixe vazio se presente_no_contrato for False.""",
    )
    motivo: str = Field(
        ...,
        description="""Por que este ponto precisa ser verificado - por exemplo, 
        'contrato não define prazo de vigência' ou 'cláusula de multa 
        presente, verificar se o percentual é abusivo'.""",
    )


class ResultadoTriagem(BaseModel):
    itens: List[ItemTriagem] = Field(
        ...,
        description="""Um item para cada uma das 8 categorias, indicando se
        está presente e o que precisa ser verificado.""",
    )
    fora_do_escopo: bool = Field(
        default=False,
        description="""True se o texto enviado não parece ser um contrato de prestação de serviço PJ 
        (ex: é sobre vínculo CLT, é sobre outro assunto completamente, ou não é um contrato).""",
    )
    observacoes_gerais: str = Field(
        default="",
        description="Observações gerais sobre o contrato como um todo, ou explicação de por que foi marcado como fora_do_escopo.",
    )


INSTRUCTIONS = [
    """
    Você é o Agente de Triagem de um sistema de análise de contratos de 
    prestação de serviço (pessoa jurídica/PJ, regidos pelo Código Civil 
    brasileiro - não contratos de trabalho CLT).,
    Sua única função é ler o contrato enviado e identificar, para cada uma 
    das 8 categorias abaixo, se ela está presente no contrato e qual 
    trecho corresponde a ela:,
    objeto, prazo, pagamento, rescisao, multa, confidencialidade, 
    propriedade_intelectual, foro.,
    Sempre devolva um item para as 8 categorias, mesmo quando a categoria 
    não aparece no contrato - nesse caso, marque presente_no_contrato como 
    False e explique em 'motivo' que a ausência da cláusula é o que 
    precisa ser avaliado.,
    Você NÃO julga se uma cláusula é boa, ruim, abusiva ou ilegal - isso é 
    trabalho de outro agente. Sua tarefa é só localizar e classificar.,
    Se o texto enviado não for um contrato de prestação de serviço PJ 
    (por exemplo, se for sobre vínculo empregatício CLT, ou não for um 
    contrato de forma alguma), marque fora_do_escopo como True e explique 
    o motivo em observacoes_gerais - não tente forçar uma categorização.,
    Nunca invente uma categoria fora das 8 listadas. """,
]

triagem_agent = Agent(
    model=Groq(
        id="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
    ),
    description=(
        "Agente de triagem que identifica quais pontos de um contrato de prestação de serviço precisam de verificação jurídica."
    ),
    instructions=INSTRUCTIONS,
    output_schema=ResultadoTriagem,
    use_json_mode=True,
)


if __name__ == "__main__":
    contrato_exemplo = """
    CONTRATO DE PRESTAÇÃO DE SERVIÇOS

    Cláusula 1ª - Objeto: O presente contrato tem por objeto a prestação de
    serviços de desenvolvimento de software pelo CONTRATADO ao CONTRATANTE.

    Cláusula 2ª - Pagamento: O CONTRATANTE pagará ao CONTRATADO o valor de
    R$ 5.000,00 mensais, até o dia 10 de cada mês.

    Cláusula 3ª - Rescisão: Este contrato poderá ser rescindido por
    qualquer das partes, sem necessidade de aviso prévio ou justificativa.
    """

    resultado = triagem_agent.run(contrato_exemplo)
    for item in resultado.content.itens:
        status = "PRESENTE" if item.presente_no_contrato else "AUSENTE"
        print(f"[{item.categoria.value.upper()}] {status}")
        print(f"  Motivo: {item.motivo}")
        if item.trecho_do_contrato:
            print(f"  Trecho: {item.trecho_do_contrato}")
        print()

    if resultado.content.fora_do_escopo:
        print("⚠️ FORA DO ESCOPO:", resultado.content.observacoes_gerais)
