import matplotlib.pyplot as plt
from pathlib import Path

# --- DADOS ---
ind_init, contra_init, ing_init = 10456, 2905, 21708
sizes_init = [ing_init, ind_init, contra_init]
labels_init = ['Ingredients', 'Indications', 'Contraindications']

ind_final, contra_final = 6568, 2383
sizes_final = [ind_final, contra_final]
labels_final = ['Indications', 'Contraindications']

# Cores consistentes
colors_init = ['#808080', '#2E86C1', '#E67E22']
colors_final = ['#2E86C1', '#E67E22']

# Caminho de saída
plots_dir = Path("data/output/plots") # Ajustado para simplicidade, podes manter o teu Path original
plots_dir.mkdir(parents=True, exist_ok=True)


# --- NOVA LÓGICA: Dois donuts no mesmo gráfico ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), facecolor='white')

# Donut inicial
wedges1, texts1, autotexts1 = ax1.pie(
    sizes_init,
    labels=labels_init,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors_init,
    pctdistance=0.85,
    wedgeprops={'width': 0.3, 'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)
ax1.text(0, 0, 'INITIAL\n35,069\nArcs', ha='center', va='center', fontsize=18, fontweight='bold')
ax1.set_aspect('equal')
ax1.set_title('Estado Inicial', fontsize=20, fontweight='bold', pad=20)

# Donut final
wedges2, texts2, autotexts2 = ax2.pie(
    sizes_final,
    labels=labels_final,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors_final,
    pctdistance=0.85,
    wedgeprops={'width': 0.3, 'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)
ax2.text(0, 0, 'FINAL\n8,951\nArcs', ha='center', va='center', fontsize=18, fontweight='bold')
ax2.set_aspect('equal')

plt.tight_layout()
output_path = plots_dir / "graph_initial_and_final_donuts.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Guardado: {output_path}")