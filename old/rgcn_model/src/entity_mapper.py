import os
import json

class EntityMapper:
    def __init__(self, data_path):
        self.data_path = data_path
        self.labels_path = os.path.join(data_path, 'clinical_labels.json')
        self.clinical_labels = {}
        
        # Carregar o mapeamento real gerado no indexing
        if os.path.exists(self.labels_path):
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                self.clinical_labels = json.load(f)

    def map_id_to_name(self, model_id):
        """Traduz o ID numérico do modelo para o nome clínico real."""
        # O JSON guarda chaves como strings, por isso convertemos o ID
        str_id = str(model_id)
        return self.clinical_labels.get(str_id, f"ID:{str_id}")

    def get_clinical_context(self, prob):
        if prob > 0.90: return "✅ INDICAÇÃO ESTABELECIDA"
        if prob > 0.70: return "🔍 CANDIDATO FORTE"
        return "❓ POTENCIAL ASSOCIAÇÃO"