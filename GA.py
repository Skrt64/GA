import numpy as np
import random

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split(',')
            diagnosis = 1 if line[1] == 'M' else 0
            features = list(map(float, line[2:]))
            data.append((diagnosis, features))
    return data

def normalize_data(data):
    max_value = max(data)
    min_value = min(data)
    
    if max_value == min_value:
        normalized_data = [0.0] * len(data)
    else:
        normalized_data = [(x - min_value) / (max_value - min_value) for x in data]
    
    return normalized_data, max_value, min_value

def denormalize_data(normalized_data, min_value, max_value):
    denormalized_data = [
        [(value * (max_value - min_value)) + min_value for value in sublist]
        for sublist in normalized_data
    ]
    return denormalized_data

def create_chromosome(input_size, hidden_size, output_size):
    num_weights = (input_size + 1) * hidden_size + (hidden_size + 1) * output_size
    return np.random.uniform(-1, 1, num_weights)

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def feed_forward(chromosome, input_data, input_size, hidden_size, output_size):
    input_hidden_weights = chromosome[:input_size * hidden_size].reshape((input_size, hidden_size))
    input_hidden_bias = chromosome[input_size * hidden_size:(input_size * hidden_size) + hidden_size]
    hidden_output_weights = chromosome[-(hidden_size * output_size):].reshape((hidden_size, output_size))
    hidden_output_bias = chromosome[-output_size:]

    hidden_layer_input = np.dot(input_data, input_hidden_weights) + input_hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, hidden_output_weights) + hidden_output_bias
    output_layer_output = sigmoid(output_layer_input)

    return output_layer_output

def fitness(chromosome, input_data, output_data, input_size, hidden_size, output_size):
    predictions = feed_forward(chromosome, input_data, input_size, hidden_size, output_size)
    mae = np.mean(np.abs(predictions - output_data))
    return 1 / (mae + 1e-6)

def stochastic_universal_sampling(population, fitness_values, num_parents):
    total_fitness = np.sum(fitness_values)
    fitness_values = fitness_values / total_fitness

    pointers = np.sort(np.random.uniform(0, 1, num_parents))
    selected_parents = []
    current_pointer = 0
    current_fitness = 0

    for i in range(len(population)):
        while current_pointer < len(pointers) and current_fitness < pointers[current_pointer]:
            selected_parents.append(population[i])
            current_pointer += 1
        if current_pointer >= len(pointers):
            break
        current_fitness += fitness_values[i]

    return np.array(selected_parents)

def tpoint_crossover(parent1, parent2, num_crossover_points):
    chromosome_length = len(parent1)
    crossover_points = np.sort(np.random.choice(chromosome_length, num_crossover_points, replace=False))
    crossover_points = np.insert(crossover_points, [0, num_crossover_points], [0, chromosome_length])

    children = []
    for i in range(num_crossover_points + 1):
        if i % 2 == 0:
            child = parent1[crossover_points[i]:crossover_points[i + 1]]
        else:
            child = parent2[crossover_points[i]:crossover_points[i + 1]]
        children.extend(child)

    return np.array(children)

def strong_mutation(chromosome, mutation_rate):
    mutated_chromosome = chromosome.copy()
    num_mutations = int(mutation_rate * len(chromosome))

    mutation_indices = np.random.choice(len(chromosome), num_mutations, replace=False)
    mutation_values = np.random.uniform(-1, 1, num_mutations)

    mutated_chromosome[mutation_indices] = mutation_values

    return mutated_chromosome

def genetic_algorithm(input_data, output_data, input_size, hidden_size, output_size, population_size, generations, num_crossover_points, mutation_rate):
    population = [create_chromosome(input_size, hidden_size, output_size) for _ in range(population_size)]
    best_fitness_history = []
    best_chromosome_history = []

    for generation in range(generations):
        fitness_values = np.array([fitness(chromosome, input_data, output_data, input_size, hidden_size, output_size) for chromosome in population])
        best_chromosome = population[np.argmax(fitness_values)]
        best_fitness = np.max(fitness_values)

        best_fitness_history.append(best_fitness)
        best_chromosome_history.append(best_chromosome)

        parents = stochastic_universal_sampling(population, fitness_values, population_size // 2)
        children = []

        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):  # ตรวจสอบขนาดของ parents
                parent1 = parents[i]
                parent2 = parents[i + 1]
                child = tpoint_crossover(parent1, parent2, num_crossover_points)
                child = strong_mutation(child, mutation_rate)
                children.append(child)

        population = children

    return best_chromosome_history, best_fitness_history

data = load_data("wdbc.data")
input_size = len(data[0][1])
hidden_size = 10
output_size = 1
population_size = random.randint(10, 100)
generations = 10
num_crossover_points = random.randint(1, 30)
mutation_rate = 0.9
random.shuffle(data)
output_data = [item[0] for item in data]
input_data = data[0][1]
normalized_input, min_input, max_input = normalize_data(input_data)

best_chromosome_history, best_fitness_history = genetic_algorithm(normalized_input, output_data, input_size, hidden_size, output_size, population_size, generations, num_crossover_points, mutation_rate)

print("best chromosome : ")
print(best_chromosome_history)
print("best fitness : ")
print(best_fitness_history)