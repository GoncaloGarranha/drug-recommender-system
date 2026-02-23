import matplotlib.pyplot as plt
from pathlib import Path

# Dados da Tabela
data = [
    ["Fármacos", "2,774", "1,524", "45.0%"],
    ["Patologias", "753", "749", "0.5%"],
    ["Total", "3,527", "2,273", "35.6%"]
]

column_labels = ["Entidade", "Antes da filtragem", "Após filtragem", "Redução"]

# Paleta de Cores baseada nos teus Donuts
color_blue = '#2E86C1'    # Cor das Indications
color_orange = '#E67E22'  # Cor das Contraindications
color_grey_bg = '#F8F9F9' # Fundo muito suave para linhas alternadas
color_border = '#D5D8DC'

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')

# Criar a tabela
table = ax.table(
    cellText=data,
    colLabels=column_labels,
    cellLoc='center',
    loc='center',
    colColours=[color_blue]*4  # Cabeçalho com o azul principal
)

# Estilização profissional
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1.2, 2.8) 

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(color_border)
    cell.set_linewidth(1)
    
    # Estilo do Cabeçalho
    if row == 0:
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
        cell.set_facecolor(color_blue)
    else:
        # Texto Geral
        cell.get_text().set_color('#2C3E50')
        
        # Destacar a coluna de Redução em negrito
        if col == 3:
            cell.get_text().set_weight('bold')
            
        # Cores alternadas nas linhas para facilitar a leitura
        if row % 2 == 0:
            cell.set_facecolor(color_grey_bg)
        else:
            cell.set_facecolor('white')

# Guardar a imagem com transparência para o PowerPoint
plots_dir = Path(__file__).parent.parent.parent / "data" / "output" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
output_path = plots_dir / "entity_reduction_table.png"

plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"Tabela guardada com sucesso em: {output_path}")