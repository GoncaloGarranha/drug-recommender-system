📂 Caracterização do Dataset de InputO Grafo de Conhecimento é alimentado por uma estrutura de dados enriquecida, composta por ficheiros em formato JSON organizados em três categorias principais, derivadas de fontes de rotulagem farmacêutica (Structured Product Labeling - SPL) e bases de dados regulatórias (como o Orange Book da FDA).🏗️ Estrutura e ProveniênciaA base de dados atual é significativamente mais robusta que as versões anteriores, apresentando uma cobertura superior de substâncias ativas e entidades patológicas:xml_files/ (~2650 ficheiros): Representam o núcleo clínico do projeto. Contêm o mapeamento direto entre fármacos e doenças através de identificadores canónicos (DOID e Orphanet). Extraem relações de indicação e contraindicação, além de listar ingredientes com códigos UNII.txt_files/ (~114 ficheiros): Focados na normalização química. Utilizam a estrutura do Orange Book para associar nomes comerciais a ingredientes ativos, normalizados via ChEBI ID.csv_files/ (~631 ficheiros): Fornecem uma camada adicional de interoperabilidade, mapeando nomes próprios e proprietários para identificadores do DrugBank.📊 Estatísticas e Riqueza do Grafo
A transição para este novo dataset representou um salto qualitativo na densidade de informação:

| Métrica | Valor Atual | Impacto |
|---------|-------------|----------|
| Fármacos Únicos (nomes) | 3.359 | Alta diversidade de nomenclatura comercial |
| Fármacos Únicos (após resolução) | 2.470 | Entidades canônicas únicas no grafo |
| DrugBank IDs | 584 | Substâncias com identificação farmacológica validada |
| Doenças (DOID) | 753 | Cobertura expandida de patologias |
| Ingredientes (UNII) | 2.021 | Componentes químicos normalizados |
| **Triplas totais (após filtragem)** | 30.659 | Base completa após blacklist |
| **Triplas preditivas (INDICATION + CONTRAINDICATION)** | 8.951 | Grafo final focado em fármaco→doença |
| Arestas de Indicação | 6.568 | Base para recomendação terapêutica (73.4%) |
| Arestas de Contraindicação | 2.383 | Camada crítica para segurança (26.6%) |
| ~~Arestas HAS_INGREDIENT~~ | ~~21.708~~ | **Removidas após análise de cold-start** |

~
******************************************************

📊 Análise de Dados: Insights e Implicações Clínicas
A auditoria inicial dos dados brutos revelou padrões críticos que moldam a estratégia de construção do Grafo de Conhecimento (KG). Abaixo, detalho as principais lições aprendidas:

1. Redundância Farmacológica (Entidades)
**O Facto:** Temos **3.359 nomes de fármacos**, mas apenas **584 IDs DrugBank únicos**.

**Implicação:** Existe uma duplicação massiva (~5.7 nomes por ID). Isto deve-se a variações de fabricantes, dosagens e nomes comerciais para a mesma substância ativa.

**Solução (Step 03 - Resolução de Entidades):** 
- **2.470 entidades canônicas** foram criadas após normalização
- **882 fármacos** resolvidos via DrugBank ID (formato: `DRUGBANK:DB00316`)
- **1.892 fármacos** resolvidos apenas por nome normalizado (formato: `DRUG:PARACETAMOL`)
- Todos os aliases (nomes comerciais, genéricos, sinônimos) são mapeados para o mesmo ID canônico
- Exemplo: "Aspirina", "Ácido Acetilsalicílico", "ASA" → todos apontam para `DRUGBANK:DB00945`
- Sem esta resolução, o modelo trataria variantes como entidades diferentes, diluindo o sinal de aprendizagem

2. Ruído e "Hub Bias" (Patologias)
**O Facto:** O termo mais frequente é o **DOID:4 ("disease")**, com **800 ocorrências**.

**Implicação:** Este é um termo ultra-genérico. Se não for removido, funcionará como um buraco negro de probabilidade (Hub), atraindo todos os fármacos para si e destruindo a precisão do ranking.

**Solução (Step 05 - Filtragem de Grafo):** 
- **Blacklist de 4 termos genéricos:**
  - `DOID:4` → "disease" (termo raiz)
  - `DOID:225` → "syndrome" (categoria ampla)
  - `DOID:162` → "cancer" (categoria ampla)
  - `DOID:848` → "arthritis" (categoria ampla)
- **1.505 triplas removidas** por conterem doenças da blacklist
- Resultado: **30.659 triplas limpas** (95.3% de retenção)
- O modelo foca-se agora em doenças específicas (ex: DOID:1240 = doença de Crohn, DOID:557 = insuficiência renal)

3. Inconsistência de Formatos (Pipeline)
**O Facto:** Os dados brutos trazem **URLs completas** (Ex: `http://purl.obolibrary.org/obo/DOID_10763`), enquanto as configurações de filtro usam o formato curto (`DOID:10763`).

**Implicação:** Qualquer filtro aplicado antes da Normalização irá falhar silenciosamente (o sistema não "vê" o DOID:4 dentro da URL longa).

**Solução (Step 02 - Parse Sources):** 
- Função `normalize_doid()` extrai e converte todos os formatos:
  - URLs → `DOID:XXXX`
  - `DOID_XXXX` → `DOID:XXXX`
  - `doid:xxxx` → `DOID:XXXX` (case-insensitive)
- Função `normalize_name()` padroniza nomes de fármacos:
  - Remove whitespace extra
  - Converte para UPPERCASE
  - Exemplo: "  Ben-u-ron  " → "BEN-U-RON"
- **Resultado:** Todas as triplas finais usam formato canônico:
  ```
  DRUG:ALLOPURINOL,INDICATION,DOID:13189
  DRUGBANK:DB00945,CONTRAINDICATION,DOID:1920
  DRUG:PARACETAMOL,HAS_INGREDIENT,UNII:362O9ITL9D
  ```

4. Escopo Ontológico e Tipos de Relações
**O Facto:** Extração de ORDO (Orphanet) com **0 ocorrências**. O foco total do dataset atual recai sobre a **Disease Ontology (DOID)**.

**Implicação:** A ontologia DOID providencia cobertura suficiente para um sistema de recomendação robusto.

**Decisão:** Manter o foco em DOID é uma escolha segura e robusta, dado que a cobertura de **753 patologias** já providencia um espaço de procura complexo o suficiente para o Mestrado.

**Tipos de Relações no Grafo:**
1. **INDICATION** (Fármaco → Doença): Relação terapêutica principal para recomendação
   - Exemplo: `DRUG:ALLOPURINOL,INDICATION,DOID:13189` (allopurinol trata gota)
   - **6.568 triplas** (73.4% das relações preditivas)
2. **CONTRAINDICATION** (Fármaco → Doença): Restrições de segurança
   - Exemplo: `DRUG:ASPIRIN,CONTRAINDICATION,DOID:12236` (aspirina contraindicada em hemofilia)
   - **2.383 triplas** (26.6% das relações preditivas)

**Total de triplas preditivas:** 8.951 (foco direto na tarefa Disease→Drug)

---

### 🔬 **Decisão: Remoção de HAS_INGREDIENT**

**Contexto inicial:**
- HAS_INGREDIENT representava 21.708 triplas (70.8% do grafo)
- 418 fármacos tinham APENAS ingredientes (sem indicações)
- Argumento inicial: cold-start via similaridade química

**Análise crítica realizada:**
```
Métricas de overlap test/train:
- Fármacos únicos no test: 552
- Fármacos cold-start no test: 2
- Fármacos "só ingredientes" no test: 0
```

**Descoberta decisiva:**
- ❌ Zero fármacos com APENAS ingredientes aparecem no test set
- ❌ Os 2 fármacos cold-start no test (ENSKYCE, JUNEL) NÃO têm ingredientes
- ❌ HAS_INGREDIENT ocupava 70.8% do grafo sem contribuir para avaliação

**Problemas de qualidade identificados:**
- Média de 11.5 ingredientes/fármaco (anormal, deveria ser 1-3)
- 2.504 pares duplicados (fármaco, ingrediente) - 11.5% das relações
- Fármacos com 92, 86, 71 ingredientes (excipientes e ruído)
- Ingrediente mais comum em 933 fármacos (excipiente, não ativo)

**Decisão fundamentada:**
1. **Não melhora a tarefa primária** (Disease→Drug no test set)
2. **Introduz ruído estrutural massivo** (70.8% de mensagens irrelevantes no R-GCN)
3. **Dilui o sinal terapêutico** (INDICATION/CONTRAINDICATION sub-representados)
4. **Simplicidade científica** (foco direto, menos variáveis a explicar)

**Resultado:** Grafo final com 8.951 triplas preditivas de alta qualidade, focado exclusivamente em relações fármaco-doença.

******************************************************