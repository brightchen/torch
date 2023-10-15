import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(color='white', linestyle='--')

plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Plot Title')

plt.xticks([0, 1, 2, 3, 4])
plt.yticks([0, 10, 20, 30, 40])
plt.grid(True)

plt.show()

#save it
#plt.savefig('my_plot.png')

