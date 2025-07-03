import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create enhanced dataset with clear patterns
np.random.seed(42)
num_samples = 200

# USN - unique identifiers
usn = ["1DS23CD" + str(i).zfill(3) for i in range(1, num_samples+1)]

# Names - random selection from given names
names = np.random.choice(['Omar', 'Maria', 'Ahmed', 'John', 'Liam', 'Sara', 'Emma', 'Ali'], size=num_samples)

# CGPA - generate with patterns based on results
cgpa = np.zeros(num_samples)
for i in range(num_samples):
    if names[i] in ['Maria', 'John']:  # Typically higher performers
        cgpa[i] = round(np.random.normal(8.5, 0.5), 2)
    else:
        cgpa[i] = round(np.random.normal(7.5, 0.8), 2)
    cgpa[i] = min(10, max(6, cgpa[i]))  # Ensure within 6-10 range

# Post graduation plans - correlated with final results
pg_plans = []
for i in range(num_samples):
    if names[i] in ['Maria', 'John']:
        pg_plans.append(np.random.choice(['Higher Studies', 'Placement'], p=[0.7, 0.3]))
    else:
        pg_plans.append(np.random.choice(['Startup', 'Placement', 'Higher Studies'], p=[0.4, 0.4, 0.2]))

# Co-curricular activities - more activities for higher performers
cocurricular = np.random.randint(1, 6, size=num_samples)
cocurricular = [x + 2 if names[i] in ['Maria', 'John'] else x for i, x in enumerate(cocurricular)]

# Recognitions - more for higher performers
recognitions = np.random.randint(0, 4, size=num_samples)
recognitions = [x + 1 if names[i] in ['Maria', 'John'] else x for i, x in enumerate(recognitions)]

# Academic Conferences/Workshops - more for those going to higher studies
academic_conf = np.random.randint(0, 5, size=num_samples)
academic_conf = [x + 2 if pg_plans[i] == 'Higher Studies' else x for i, x in enumerate(academic_conf)]

# Entrepreneurship experience - more for startup track
entrepreneurship = np.random.randint(0, 4, size=num_samples)
entrepreneurship = [x + 2 if pg_plans[i] == 'Startup' else x for i, x in enumerate(entrepreneurship)]

# Skills for startup - more for startup track
startup_skills = np.random.randint(1, 6, size=num_samples)
startup_skills = [x + 2 if pg_plans[i] == 'Startup' else x for i, x in enumerate(startup_skills)]

# Programming languages - more for placement/higher studies
prog_langs = np.random.randint(1, 6, size=num_samples)
prog_langs = [x + 1 if pg_plans[i] in ['Placement', 'Higher Studies'] else x for i, x in enumerate(prog_langs)]

# Coding - more for placement/higher studies
coding = np.random.randint(1, 6, size=num_samples)
coding = [x + 1 if pg_plans[i] in ['Placement', 'Higher Studies'] else x for i, x in enumerate(coding)]

# Internship - more for placement track
internship = np.random.randint(0, 4, size=num_samples)
internship = [x + 2 if pg_plans[i] == 'Placement' else x for i, x in enumerate(internship)]

# Research papers - more for higher studies
research_papers = np.random.randint(0, 4, size=num_samples)
research_papers = [x + 2 if pg_plans[i] == 'Higher Studies' else x for i, x in enumerate(research_papers)]

# Results - our target variable
results = []
for i in range(num_samples):
    if pg_plans[i] == 'Higher Studies':
        results.append('Higher Studies')
    elif pg_plans[i] == 'Startup':
        results.append('Startup')
    else:
        results.append('Placement')

# Create DataFrame
data = pd.DataFrame({
    'usn': usn,
    'name': names,
    'cgpa': cgpa,
    'post graduation plans': pg_plans,
    'cocurricular activities': cocurricular,
    'recognitions': recognitions,
    'Academic Conferences/Workshops': academic_conf,
    'enterpreunership experience': entrepreneurship,
    'skills for stratup': startup_skills,
    'programming languages': prog_langs,
    'coding': coding,
    'internship': internship,
    'research papers': research_papers,
    'Results': results
})

# Fix typo in "Placement" (original had "Plaecment")
data['Results'] = data['Results'].replace('Placement', 'Plaecment')

# Verify the patterns
print(data.groupby('name')['Results'].value_counts(normalize=True))
print("\nFeature correlations with Results:")
print(data.corrwith(pd.factorize(data['Results'])[0]))

# Split data for testing
X = data.drop(['usn', 'name', 'Results', 'post graduation plans'], axis=1)
y = data['Results']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
preds = rf.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, preds):.2%}")