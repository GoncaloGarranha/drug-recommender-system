📂 Caracterização do Dataset de InputO Grafo de Conhecimento é alimentado por uma estrutura de dados enriquecida, composta por ficheiros em formato JSON organizados em três categorias principais, derivadas de fontes de rotulagem farmacêutica (Structured Product Labeling - SPL) e bases de dados regulatórias (como o Orange Book da FDA).🏗️ Estrutura e ProveniênciaA base de dados atual é significativamente mais robusta que as versões anteriores, apresentando uma cobertura superior de substâncias ativas e entidades patológicas:xml_files/ (~2650 ficheiros): Representam o núcleo clínico do projeto. Contêm o mapeamento direto entre fármacos e doenças através de identificadores canónicos (DOID e Orphanet). Extraem relações de indicação e contraindicação, além de listar ingredientes com códigos UNII.txt_files/ (~114 ficheiros): Focados na normalização química. Utilizam a estrutura do Orange Book para associar nomes comerciais a ingredientes ativos, normalizados via ChEBI ID.csv_files/ (~631 ficheiros): Fornecem uma camada adicional de interoperabilidade, mapeando nomes próprios e proprietários para identificadores do DrugBank.📊 Estatísticas e Riqueza do GrafoA transição para este novo dataset representou um salto qualitativo na densidade de informação:MétricaValor AtualImpactoFármacos Únicos~2.650Aumento de ~75% na diversidade molecular.Doenças (DOID)~705Cobertura expandida de patologias.Arestas de Indicação7.551Base para o treino de recomendação terapêutica.Arestas de Contraindicação2.905Camada crítica para filtragem de segurança.~
******************************************************

📊 Análise de Dados: Insights e Implicações Clínicas
A auditoria inicial dos dados brutos revelou padrões críticos que moldam a estratégia de construção do Grafo de Conhecimento (KG). Abaixo, detalho as principais lições aprendidas:

1. Redundância Farmacológica (Entidades)
O Facto: Temos 3359 nomes de fármacos, mas apenas 584 IDs DrugBank únicos.

Implicação: Existe uma duplicação massiva (~5.7 nomes por ID). Isto deve-se a variações de fabricantes, dosagens e nomes comerciais para a mesma substância ativa.

Solução (Step 03): A Resolução de Entidades é obrigatória. Sem ela, o modelo trataria "Aspirina" e "Ácido Acetilsalicílico" como coisas diferentes, diluindo o sinal de aprendizagem.

2. Ruído e "Hub Bias" (Patologias)
O Facto: O termo mais frequente é o DOID_4 ("disease"), com 800 ocorrências.

Implicação: Este é um termo ultra-genérico. Se não for removido, funcionará como um buraco negro de probabilidade (Hub), atraindo todos os fármacos para si e destruindo a precisão do ranking.

Solução (Step 05): Filtragem cirúrgica de hubs genéricos. O modelo deve focar-se em doenças específicas (ex: Asma, Diabetes) para ter utilidade clínica.

3. Inconsistência de Formatos (Pipeline)
O Facto: Os dados brutos trazem URLs (Ex: .../DOID_10763), enquanto as configurações de filtro usam o formato curto (DOID:10763).

Implicação: Qualquer filtro aplicado antes da Normalização irá falhar silenciosamente (o sistema não "vê" o DOID_4 dentro da URL longa).

Solução (Step 02/04): Uniformização obrigatória de strings para o formato canónico DOID:XXXX antes da construção das triplas.

4. Escopo Ontológico
O Facto: Extração de ORDO (Orphanet) com 0 ocorrências.

Implicação: O foco total do dataset atual recai sobre a Disease Ontology (DOID).

Decisão: Manter o foco em DOID é uma escolha segura e robusta, dado que a cobertura de ~750 patologias já providencia um espaço de procura complexo o suficiente para o Mestrado.

******************************************************