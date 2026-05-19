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

*** 🔄 Por que é que recuar agora é o passo mais inteligente da Tese? ***
No desenvolvimento de sistemas de Inteligência Artificial, existe um princípio sagrado: "Garbage In, Garbage Out" (Lixo entra, Lixo sai). Se os alicerces de dados são frágeis, o modelo mais sofisticado do mundo — como o teu R-GCN — torna-se apenas uma "caixa preta" que cospe números sem significado.

Recuar para o kg_constructor não é um atraso; é uma recalibragem estratégica por três razões fundamentais:

A Preservação da "Linhagem de Dados": Até agora, estávamos a tentar "adivinhar" nomes que o sistema já conhecia no momento da extração. É como traduzir um livro para uma língua que não dominamos e depois tentar traduzi-lo de volta com um dicionário incompleto. Ao voltarmos à extração, garantimos que cada entidade nasce com a sua identidade (o nome clínico) colada ao seu BI (o ID técnico).

Soberania e Independência Semântica: Depender de APIs externas (FDA, OLS) para identificar o que está no nosso próprio grafo é um risco académico. Se a API estiver offline ou o ID for antigo, o teu modelo fica "mudo". Ao integrarmos os nomes diretamente no construtor, o teu sistema torna-se autossuficiente e robusto.

Interpretabilidade: O Coração de uma Tese de Mestrado: Um júri de mestrado não avalia apenas se o modelo "acerta"; avalia se tu sabes porque é que ele acerta. Se o modelo prevê que o fármaco A é indicado para o nó 8893, e tu não consegues dizer imediatamente que o 8893 é "Asma", a validade científica do trabalho é posta em causa. Recuamos para garantir que cada predição seja uma frase legível e não um enigma matemático.

*** Para mitigar a redundância semântica causada pela co-existência de nomes comerciais e genéricos nas bulas da FDA, implementou-se uma camada de Resolução de Entidades (Entity Resolution). Utilizou-se uma hierarquia de normalização que prioriza o identificador universal ChEBI, fundindo múltiplas instâncias de produtos na sua substância ativa canónica. Esta estratégia aumentou a densidade do grafo e evitou a dispersão de gradientes durante o treino da rede R-GCN.

*** 📝 Notas de Desenvolvimento: Refinação da Ontologia e Pipeline DGL ***
Contexto: Após os testes preliminares com o modelo R-GCN, observou-se que a inferência clínica, embora funcional, apresentava ruído semântico significativo (ex: associações a termos genéricos como "Disease" e confusão entre marcas comerciais e substâncias genéricas).

Decisão de "Backtracking": Decidiu-se retroceder no pipeline de dados para a fase de extração pós-MedJsonify pelas seguintes razões:

Resolução de Entidades (Entity Resolution): A presença de múltiplos nós para a mesma substância ativa (ex: Advil, Motrin, Ibuprofeno) dispersava a densidade do grafo. Implementou-se uma hierarquia de normalização baseada no ChEBI ID como identificador canónico, fundindo nomes comerciais na sua substância base.

Mitigação de Ruído Estrutural: Identificou-se que nós de topo da Disease Ontology (como DOID:4 - Disease) funcionavam como "hubs" artificiais, enviesando as predições. Criou-se uma blacklist de exclusão para estes termos genéricos.

Adoção de Padrões de Estado-da-Arte (DGL): Para garantir a defesa académica da tese, optou-se por migrar a reimplementação para a biblioteca DGL (Deep Graph Library). Esta mudança permite:

Adotar uma reimplementação benchmark de R-GCN + DistMult inspirada no paper original (Schlichtkrull et al.).

Utilizar um descodificador DistMult para Link Prediction.

Garantir a reprodutibilidade através de métricas padrão da literatura (MRR e Hits@K).

Objetivo Final: Transformar um protótipo de extração textual num Grafo de Conhecimento Clínico estruturado, onde a topologia do grafo reflita relações biológicas reais e não apenas co-ocorrências estatísticas de palavras em bulas.

*** A execução do novo pipeline de extração resultou numa redução de 21% no volume de triplos (de 7059 para 5581), confirmando a eficácia da estratégia de deduplicação canónica (ChEBI/Genéricos). A consolidação de múltiplas marcas comerciais num único nó substância permitiu aumentar a densidade informativa do grafo, enquanto a exclusão seletiva de 5 nós estruturais (hubs genéricos) eliminou 'atalhos' semânticos que prejudicavam a precisão da inferência.

*** A análise estatística pós-purificação revelou um grafo composto por 5.581 triplos únicos, conectando 1.505 fármacos a 703 patologias distintas. A consolidação de entidades resultou num rácio de conectividade média de 3.7 arestas por nó de fármaco. A verificação da distribuição de grau em escala logarítmica confirmou a natureza topológica da rede, validando a sua adequação para o uso de algoritmos de Message Passing como a R-GCN.

*** 📝 Caracterização Estatística do Grafo de Conhecimento (EDA)  ***
Contexto e Dimensão: Após a fase de extração e normalização canónica, o Grafo de Conhecimento (KG) consolidou-se com 5.581 triplos únicos, estabelecendo relações entre 1.505 fármacos e 703 patologias. A manutenção de uma única relação semântica (indicated_for) nesta fase assegura um foco rigoroso na tarefa de recomendação terapêutica.

Análise de Conetividade (Degree Distribution): A estrutura apresenta uma conetividade média de 3,7 indicações por fármaco, com uma mediana de 2,0. A análise dos percentis revela uma distribuição de "cauda longa" (long-tail distribution):

P90 (7,0): 90% dos fármacos possuem até 7 indicações.

P99 (19,0): Existe uma elite de fármacos altamente versáteis.

Máximo (78): A presença de hubs (nós altamente conetados) é fundamental para a propagação de mensagens (message passing) na R-GCN, pois estes nós funcionam como pontes de conhecimento entre diferentes áreas da ontologia.

Validação de Qualidade: A redução do volume total de triplos (em comparação com a extração bruta) sem perda de diversidade de entidades confirma a eficácia da Deduplicação Canónica e da Filtragem de Nós Genéricos. O grafo resultante possui uma topologia "Scale-Free" típica de redes biológicas reais, o que valida a sua robustez para o treino de modelos de Graph Neural Networks.