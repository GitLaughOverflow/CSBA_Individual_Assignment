import numpy as np
import matplotlib.pyplot as plt
import string
import json 
import re
import math
import difflib

from itertools import chain
from itertools import combinations

class Product:
    def __init__(self, title, P1, P2, shop, modelID):
        self.title = title
        self.P1  = P1
        self.P2  = P2
        self.shop = shop
        self.modelID = modelID
        def title_maker(text):
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = text.translate(translator)
    cleaned_text = cleaned_text.lower()
    return cleaned_text


def P1_maker(string, k=5):
    shingles = []
    text = string.replace(" ", "")
    for i in range(0, len(text) - k + 1):
        shingles.append(text[i:(i+k)])
    return list(set(shingles))


def P2_maker(string):
    words = string.lower().split()
    alphanumeric_words = []
    
    for word in words:
        if re.search(r'\d', word) and re.search(r'[a-zA-Z]', word):
            cleaned_word = ''.join(char for char in word if char.isalnum() or char.isspace())
            alphanumeric_words.append(cleaned_word)
            
            
    for word in alphanumeric_words:
        if "inch" in word:
            alphanumeric_words.remove(word)
            
    for word in alphanumeric_words:
        if word == "3d":
            alphanumeric_words.remove(word)
            
    
    if len(alphanumeric_words) == 0:
        alphanumeric_words.append('no_model_words')
        
    return list(set(alphanumeric_words))

def multiply_rows_by_power(arr):
    num_rows = arr.shape[0] 
    powers = np.power(100, np.arange(num_rows))  
    multiplied = arr * powers[:, np.newaxis]
    column_sums = np.sum(multiplied, axis=0)   
    return column_sums


def step2(check):
    double_check = []
    for pair in check:
        if compare(pair[0].P2, pair[1].P2) > 0.4:
            double_check.append(pair)
    return double_check

def compare(arr1, arr2):
    # Concatenate array elements into strings
    arr1_str = ' '.join(str(elem) for elem in arr1)
    arr2_str = ' '.join(str(elem) for elem in arr2)

    # Use difflib to compare the strings representing the arrays
    similarity_ratio = difflib.SequenceMatcher(None, arr1_str, arr2_str).ratio()
    return similarity_ratio



  with open("TVs-all-merged.json", 'r') as json_file:
    data = json.load(json_file)
    
Product_List = []
duplicate_count = 0
count = 1
for i in data:
    for j in data[i]:
        title = title_maker(j['title'])
        item = Product(title, P1_maker(title), P2_maker(title), j['shop'], j['modelID'])
        
        if len(data[i]) > 1:
            item.duplicate = count
            duplicate_count += 1
            
        else:
            item.duplicate = 0
        
        Product_List.append(item)
     
    if len(data[i]) > 1:
        count += 1


        def universal_set(Product_List):
    universal_set_P1 = []
    universal_set_P2 = []
    for product in Product_List:
        for i in product.P1:
            universal_set_P1.append(i)
        for j in product.P2:
            universal_set_P2.append(j)     
    return list(set(universal_set_P1)), list(set(universal_set_P2))


    def input_matrix(u_set_P1, u_set_P2):
    store_P1 = []
    store_P2 = []

    for product in Product_List:
        sparse_vector_P1 = np.zeros(len(u_set_P1))
        for j in range(len(u_set_P1)):
            if u_set_P1[j] in product.P1:
                sparse_vector_P1[j] = 1
        store_P1.append(sparse_vector_P1)

        sparse_vector_P2 = np.zeros(len(u_set_P2))
        for z in range(len(u_set_P2)):
            if u_set_P2[z] in product.P2:
                sparse_vector_P2[z] = 1
        store_P2.append(sparse_vector_P2)

    input_matrix_P1 = np.array(store_P1).T
    input_matrix_P2 = np.array(store_P2).T
    
    return input_matrix_P1, input_matrix_P2


    u_set_P1 = universal_set(Product_List)[0]
u_set_P2 = universal_set(Product_List)[1]

I1, I2 = input_matrix(u_set_P1, u_set_P2)

def create_SignatureMatrix(input_matrix, num_hashes):
    np.random.seed(2)
    B = 379
    a = np.random.randint(low=1, high=100000, size=num_hashes)
    b = np.random.randint(low=1, high=100000, size=num_hashes)

    hashmatrix = np.zeros((len(input_matrix), num_hashes))

    for i in range(len(input_matrix)):
        for j in range(num_hashes):
            hashmatrix[i,j] = (a[j] * i + b[j]) % B
            
    signature = np.full((num_hashes, len(input_matrix[0])), np.inf)

    for i in range(len(input_matrix)):
        for j in range(len(input_matrix[0])):
            if input_matrix[i,j] == 1:
                v1 = signature[:,j]
                v2 = hashmatrix[i,:]
                signature[:,j] = np.minimum(v1, v2)

    return signature


    def return_combinations(signature, num_bands, num_buckets, product_list):
    store = []
    for band in np.split(signature,num_bands):
        store.append(multiply_rows_by_power(band))
    matrix = np.array(store)

    num_buckets = num_buckets
    total_list = []
    for row in matrix:
        buckets = [[] for i in range(num_buckets)]
        for element in range(len(row)):
            buckets[int(row[element] % num_buckets)].append(product_list[element])
        total_list.append(buckets)

    flattened_list = list(chain.from_iterable(total_list))
    filtered_list = [sublist for sublist in flattened_list if len(sublist) >= 2]


    flagged_combinations = []
    for i in filtered_list:
        for j in combinations(i,2):
            flagged_combinations.append(j)

    unique_combinatons = list(set(flagged_combinations))
    
    return unique_combinatons

    def performance(check, Product_List):
    
    
    true_duplicates = [obj for obj in Product_List if obj.duplicate > 0]
    true_duplicates_lists = [[] for i in range(330-1)]
    true_duplicates_combinations = []

    for i in range(330-1):
        for j in true_duplicates:
            if j.duplicate == (i+1):
                true_duplicates_lists[i].append(j) 

    for sublist in true_duplicates_lists:
        for combi in combinations(sublist,2):
            true_duplicates_combinations.append(combi)
            
    true_duplicates_combinations = list(set(true_duplicates_combinations))
    
    TP = FP = FN = 0 

    for pair in check:
        if pair[0].modelID == pair[1].modelID:
            TP += 1       
        else: 
            FP += 1

    for pair in true_duplicates_combinations:
        if pair not in check:
            FN += 1
            
    
    Precision = TP / (TP + FP)
    Recall  = TP / (TP + FN)
    
    
    
    
    pair_quality = TP / len(check)
    
    pair_completeness =  TP / len(true_duplicates_combinations)
    
    F1  = (2 * (Precision * Recall) / (Precision + Recall)) 
    
    x_value = len(check) / math.comb(1624, 2)
    
    return x_value, F1, pair_quality, pair_completeness

def performance2(check, orginal_check):
    
    flattened_list = list(set([item for tup in orginal_check for item in tup]))
    
    true_duplicates = [obj for obj in flattened_list if obj.duplicate > 0]
    true_duplicates_lists = [[] for i in range(330-1)]
    true_duplicates_combinations = []

    for i in range(330-1):
        for j in true_duplicates:
            if j.duplicate == (i+1):
                true_duplicates_lists[i].append(j) 

    for sublist in true_duplicates_lists:
        for combi in combinations(sublist,2):
            true_duplicates_combinations.append(combi)
            
    true_duplicates_combinations = list(set(true_duplicates_combinations))
    
    TP = FP = FN = 0 

    for pair in check:
        if pair[0].modelID == pair[1].modelID:
            TP += 1       
        else: 
            FP += 1

    for pair in true_duplicates_combinations:
        if pair not in check:
            FN += 1
            
    
    Precision = TP / (TP + FP)
    Recall  = TP / (TP + FN)
    
    
    pair_quality = TP / len(check)
    
    pair_completeness =  TP / len(true_duplicates_combinations)
    
    F1  = (2 * (Precision * Recall) / (Precision + Recall)) 
    
    x_value = len(check) / math.comb(1624, 2)
    
    return x_value, F1, pair_quality, pair_completeness


    def plot(x_store,F1_store, pair_quality, pair_completeness, file_name):
    plt.figure()
    plt.plot(x_store, F1_store)
    plt.ylabel('F1')
    plt.xlabel('Fraction of Comparisons')
    plt.title('Performance Measure: F1')
    plt.savefig(f"{file_name}_F1", dpi=300)

    plt.figure()
    plt.plot(x_store, pair_quality)
    plt.ylabel('Pair Quality')
    plt.xlabel('Fraction of Comparisons')
    plt.title('Performance Measure: Pair Quality')
    plt.savefig(f"{file_name}_PQ", dpi=300)

    plt.figure()
    plt.plot(x_store, pair_completeness)
    plt.ylabel('Pair Completeness')
    plt.xlabel('Fraction of Comparisons')
    plt.title('Performance Measure: Pair Completeness')
    plt.savefig(f"{file_name}_PC", dpi=300)


    x_store1 = []
F1_store1 = []
pair_quality1 = []
pair_completeness1 = []


x_store2 = []
F1_store2 = []
pair_quality2 = []
pair_completeness2 = []

for i in [500, 250, 200, 125, 100, 50, 25, 20, 10, 5, 4, 2]:
    check1 = return_combinations(S1, i, 16000, Product_List)
    result1 = performance(check1, Product_List)
    x_store1.append(result1[0])
    F1_store1.append(result1[1])
    pair_quality1.append(result1[2])
    pair_completeness1.append(result1[3])
    
    
    double_check = step2(check1)
    
    result2 = performance2(double_check, check1)
    x_store2.append(result2[0])
    F1_store2.append(result2[1])
    pair_quality2.append(result2[2])
    pair_completeness2.append(result2[3])
    
    
