import csv
import matplotlib.pyplot as plt

angles = []
powers = []

with open('SHG.csv', 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        angles.append(float(row['Angle']))
        powers.append(float(row['Power']))

plt.figure(figsize=(8, 5))
plt.plot(angles, powers, marker='o')
plt.xlabel('Angle (degrees)')
plt.ylabel('Power')
plt.title('SHG Power vs Angle')
plt.grid(True)
plt.tight_layout()
plt.show()