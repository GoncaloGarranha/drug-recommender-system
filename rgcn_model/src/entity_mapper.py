import os
import pandas as pd

class EntityMapper:
    """
    Classe responsável por mediar a tradução entre identificadores técnicos 
    do Grafo de Conhecimento e terminologia clínica legível.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        # Dicionário de tradução manual para conceitos detetados no treino
        # Podes ir alimentando esta lista à medida que descobres novos códigos
        self.clinical_dictionary = {
            "1389": "Respiratory System Context",
            "31174": "Bronchospasm / Asthma Symptoms",
            "10763": "Chronic Obstructive Pulmonary Disease (COPD)",
            "4500": "Palivizumab (Anticorpo VSR)",
            "1826": "Epileptic Seizure Control",
            "3238": "Fluoxetine (Antidepressant)",
            # Adiciona mais aqui...
        }

    def map_id_to_name(self, entity_id):
        """Traduz um ID para um nome humano usando múltiplas estratégias."""
        
        # Estratégia 1: Verificação no Dicionário Manual
        if entity_id in self.clinical_dictionary:
            return self.clinical_dictionary[entity_id]

        # Estratégia 2: Heurística para Nomes Legíveis (Fármacos já em texto)
        if not entity_id.isdigit() and "-" not in entity_id:
            return entity_id # Já é um nome (ex: "ALBUTEROL")

        # Estratégia 3: Identificação de Metadados FDA (SetIDs)
        if "-" in entity_id and len(entity_id) > 20:
            return f"FDA_Document_Ref_{entity_id[:8]}"

        # Estratégia 4: Fallback para IDs numéricos desconhecidos
        return f"Medical_Concept_ID:{entity_id}"

    def get_clinical_context(self, prob):
        """Retorna uma etiqueta visual baseada na probabilidade."""
        if prob > 0.95: return "✅ INDICAÇÃO ESTABELECIDA"
        if prob > 0.80: return "🔍 CANDIDATO FORTE"
        return "❓ POTENCIAL ASSOCIAÇÃO"