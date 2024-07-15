import random
import numpy as np
import matplotlib.pyplot as plt

# تعداد شهرها
num_cities = 6

# ایجاد موقعیت‌های تصادفی برای شهرها
cities = np.random.rand(num_cities, 2)

# محاسبه فاصله بین دو شهر و گرد کردن آن
def distance(city1, city2):
    return round(np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2))

# محاسبه طول یک مسیر و گرد کردن آن
def path_length(path, cities):
    return round(sum(distance(cities[path[i]], cities[path[i+1]]) for i in range(len(path)-1)) + distance(cities[path[-1]], cities[path[0]]))

# تولید یک جمعیت اولیه
def create_initial_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        path = list(np.random.permutation(num_cities))
        population.append(path)
    return population

# انتخاب افراد برای تولید نسل بعدی
def selection(population, cities, num_selected):
    fitness_scores = [(path, 1 / path_length(path, cities)) for path in population]
    fitness_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [path for path, _ in fitness_scores[:num_selected]]
    return selected

# ترکیب دو مسیر برای تولید فرزندان
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end+1] = parent1[start:end+1]
    pointer = 0
    for city in parent2:
        if city not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = city
    return child

# جهش در یک مسیر
def mutate(path, mutation_rate):
    for i in range(len(path)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(path)-1)
            path[i], path[j] = path[j], path[i]

# الگوریتم ژنتیک برای حل مسئله فروشنده دوره گرد
def genetic_algorithm(cities, pop_size, num_generations, mutation_rate):
    population = create_initial_population(pop_size, len(cities))
    for generation in range(num_generations):
        selected = selection(population, cities, pop_size // 2)
        offspring = []
        while len(offspring) < pop_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            offspring.append(child)
        population = offspring
    best_path = min(population, key=lambda path: path_length(path, cities))
    return best_path

# پارامترها
pop_size = 200
num_generations = 1000
mutation_rate = 0.02

# اجرای الگوریتم
best_path = genetic_algorithm(cities, pop_size, num_generations, mutation_rate)



print ("-----------------------------------")
print("city's number: ",num_cities)
print("num_generations : ",num_generations)
print("mutation_rate: ",mutation_rate)
print ("-----------------------------------")
# نمایش طول بهترین مسیر به صورت عدد صحیح
print("Best path length:", path_length(best_path, cities))
print ("___________________________________")




# نمایش بهترین مسیر
plt.figure(figsize=(10, 6))
for i in range(len(best_path)):
    start_city = cities[best_path[i]]
    end_city = cities[best_path[(i+1) % len(best_path)]]
    plt.plot([start_city[0], end_city[0]], [start_city[1], end_city[1]], 'bo-')
plt.show()

