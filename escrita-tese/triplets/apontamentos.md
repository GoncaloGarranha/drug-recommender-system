*** No teu Capítulo 4 (Aprendizagem e Modelação), deves escrever algo como: ***

"Embora os dados sejam processados em formato de tripletos seguindo a lógica semântica do modelo RDF (Resource Description Framework), a arquitetura R-GCN é utilizada para computar representações latentes que preservam a distinção entre as diversas tipologias de relações. Isto permite que o modelo aprenda que a influência de uma 'substância ativa' na representação de um fármaco é distinta da influência de uma 'indicação terapêutica'."

***Taxonomia de Fontes e Mapeamento Polimórfico***
Dada a elevada heterogeneidade dos dados disponibilizados pela FDA, a construção do Grafo de Conhecimento exigiu uma abordagem de engenharia de dados flexível, capaz de processar diferentes esquemas de metadados sem comprometer a integridade semântica dos tripletos resultantes. Para tal, o módulo kg_constructor utiliza um motor de mapeamento configurável que categoriza as fontes em quatro "famílias" principais, baseadas na assinatura estrutural dos ficheiros JSON processados:

Rótulos Clínicos Estruturados (Família XML): Esta categoria engloba os registos mais complexos, onde a presença da chave indications serve como detetor de assinatura. Estes documentos possuem uma estrutura profunda, permitindo a extração de relações ricas como indicated_for e contraindicated_for, utilizando o identificador universal id (UUID) como sujeito do triplo.

Registos Tabulares de Pequenas Moléculas (Orange Book): Identificada pela chave Trade_Name, esta família provém de dados originalmente tabulares (TXT). O mapeamento foca-se na transposição de nomes comerciais para ingredientes ativos, estabelecendo a base da conectividade química do grafo.

Registos de Produtos Biológicos (Purple Book): Semelhante à anterior, mas detetada pela chave Proprietary Name. Esta distinção é crucial, pois os produtos biológicos possuem terminologias específicas (e.g., Proper Name) que requerem regras de normalização distintas para garantir a correta indexação das entidades.

Rótulos Simplificados e Suplementos (Família Ingredients): Identificada durante a fase de validação, esta família agrupa produtos com documentação simplificada, como gases medicinais (Oxigénio, Hélio) e suplementos vitamínicos. Estes ficheiros carecem de secções narrativas de indicações, sendo a extração baseada na chave de topo ingredients. A inclusão desta regra permitiu integrar no grafo entidades fundamentais que, de outra forma, seriam descartadas por falta de metadados clínicos complexos.

*** Rastreio de Volumetria e Eficiência de Extração ***
Após a consolidação das regras de mapeamento para as quatro famílias de dados identificadas, procedeu-se à extração sistemática de todo o repositório de entidades processadas pelo MedJsonify. Este processo permitiu o rastreio exato da proveniência dos dados, garantindo a integridade da data lineage do projeto.

O inventário final de ficheiros de entrada e o resultado da extração apresentam a seguinte distribuição:

Fontes Processadas (Total: 3.395 ficheiros):

XML (Clinical Labels): 2.650 ficheiros (o núcleo denso de informação terapêutica).

CSV (Purple Book): 631 ficheiros (focados em produtos biológicos).

TXT (Orange Book): 114 ficheiros (focados em pequenas moléculas e patentes).

Resultado da Extração: A execução do motor de mapeamento resultou na geração de 11.938 tripletos únicos.

Análise de Rendimento: Observa-se um rácio médio de aproximadamente 3,5 relações por documento. Este valor é considerado ideal, pois demonstra que o sistema foi capaz de filtrar o ruído textual das bulas (como instruções de embalagem ou descrições administrativas) e extrair apenas os factos semânticos relevantes para o Grafo de Conhecimento. O facto de termos quase 12.000 factos extraídos de cerca de 3.400 documentos garante uma densidade relacional suficiente para que o modelo R-GCN consiga aprender representações significativas das entidades farmacológicas e patológicas.

*** Indexação e Particionamento Estratificado ***

Para converter o Grafo de Conhecimento num formato compatível com arquiteturas de Deep Learning (especificamente R-GCN), procedeu-se à indexação numérica de todas as entidades e relações. Este processo resultou num dicionário de 4.637 entidades únicas e 6 categorias de relações.

O dataset foi dividido seguindo as melhores práticas de aprendizagem automática:

Treino (80% - 9.550 triplos): Utilizado para a atualização dos pesos da rede e aprendizagem dos embeddings relacionais.

Validação (10% - 1.194 triplos): Destinado ao ajuste de hiperparâmetros e prevenção de overfitting.

Teste (10% - 1.194 triplos): Reservado exclusivamente para a avaliação final do desempenho do modelo em tarefas de Link Prediction.

A divisão foi realizada através de amostragem estratificada, garantindo que a proporção original das relações (ex: a raridade de has_organization vs. a abundância de indicated_for) se mantém consistente em todos os conjuntos de dados.