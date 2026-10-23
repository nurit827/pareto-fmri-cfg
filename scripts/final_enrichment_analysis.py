import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# טעינת הנתונים
features = pd.read_csv('data/features.csv')
behavior = pd.read_csv('data/continuous_enrichment.csv')

# מיזוג לפי עמודת ID
common_id = 'id' if 'id' in features.columns else features.columns[0]
df = pd.merge(features, behavior, left_on=common_id, right_on=behavior.columns[0])

# בחירת הארכיטיפים (נניח 3 עמודות ראשונות אחרי ה-ID)
arch_cols = [c for c in features.columns if c != common_id][:3]
# בחירת משתני התנהגות מהרשימה שראינו קודם
target_cols = ['Age_in_Yrs', 'ASR_Totp_T', 'DSM_Depr_T', 'PsychWellBeingMean'] 
target_cols = [c for c in target_cols if c in behavior.columns]

# חישוב מתאמים
results = pd.DataFrame(index=target_cols, columns=arch_cols)
for arch in arch_cols:
    for target in target_cols:
        r, p = spearmanr(df[arch], df[target], nan_policy='omit')
        results.loc[target, arch] = r

# יצירת הגרף
plt.figure(figsize=(10, 8))
sns.heatmap(results.astype(float), annot=True, cmap='RdBu_r', center=0)
plt.title('Archetype Behavioral Enrichment')
plt.savefig('results/final_enrichment_plot.pdf')
print("--- SUCCESS: Graph saved as results/final_enrichment_plot.pdf ---")
