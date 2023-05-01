import json
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, TFBertModel
import numpy as np
from nltk.corpus import wordnet
from sklearn.ensemble import RandomForestClassifier


#  List Of Unique Symptom Tokens
def UniqueTokens():
    # Load Dataset
    with open('Dataset.json') as f:
        data = json.load(f)
    
    unique_symptoms = set()
    stop_words = set(stopwords.words('english'))
    tokenizer = nltk.RegexpTokenizer(r'\w+')
           
    for disease in data['data']:
        symptoms = disease['symptoms']
        symptoms = symptoms.lower()
        symptoms = [word for word in tokenizer.tokenize(symptoms) if word.lower() not in stop_words]
        
        for symptom in symptoms:
            if symptom not in stop_words:
                unique_symptoms.add(symptom)
    
    return list(unique_symptoms)


# Returns Nearest Synonym From List Of Synonym
def Synonym(word, words):
    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    synset = synsets[0]
    synonyms = set(synset.lemma_names())
    nearest_synonym = None
    max_similarity = -1
    for w in words:
        w_synsets = wordnet.synsets(w)
        if not w_synsets:
            continue
        w_synset = w_synsets[0]
        w_synonyms = set(w_synset.lemma_names())
        similarity = synset.path_similarity(w_synset)
        if similarity is not None and similarity > max_similarity and w_synonyms.intersection(synonyms):
            max_similarity = similarity
            nearest_synonym = w
    return nearest_synonym





def PredictDisease(Input: str) -> str:


    # Basic Deduction That Are Rule Based And Require No AI & ML
    Input_list = Input.lower().split(",")
    if len(Input_list) == 2:
        if ('fever' in Input_list) and ("cough" in Input_list):
            return 'viral fever'
    elif len(Input_list) == 1:
        if 'fever' in Input_list:
            return 'normal fever'
        elif 'cough' in Input_list:
            return 'cough'
        elif 'nose' in Input_list[0]:
            return 'common cold'
        
    # Load Dataset
    with open('Dataset.json') as f:
        dataset = json.load(f)
        
    # Tokenize and Remove Stop Words Using NLTK (Initializing)
    nltk.download('punkt')
    nltk.download('stopwords')
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))

    # Symptom Dictionary: It Maps Each Symptom To The Disease It's Associated With
    symptom_dict = {}
    for data_point in dataset['data']:
        symptoms = data_point['symptoms'].lower()
        symptoms_nostop_list = [word for word in tokenizer.tokenize(symptoms) if word.lower() not in stop_words]
        for symptom in symptoms_nostop_list:
            if symptom not in symptom_dict:
                symptom_dict[symptom] = [data_point['disease']]
            else:
                symptom_dict[symptom].append(data_point['disease'])
    
    # Preprocessing Of Input
    Input = Input.lower()
    Input_list = [word for word in tokenizer.tokenize(Input) if word.lower() not in stop_words]
    Input = ' '.join(Input_list)


    # Use BERT to generate numerical representations of the symptoms
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    symptoms_input = tokenizer.encode_plus(Input, add_special_tokens=True, return_token_type_ids=False, return_attention_mask=True, padding='max_length', max_length=128, truncation=True)
    input_ids = np.array(symptoms_input['input_ids'])
    attention_mask = np.array(symptoms_input['attention_mask'])
    output = model(input_ids=tf.constant([input_ids]), attention_mask=tf.constant([attention_mask]))

    
    # Make A List Of Diseases
    diseases_one = []
    Symptom_list = Input.split()

    for sym_point in Symptom_list:
        if sym_point in symptom_dict:
            l = symptom_dict[sym_point]
            diseases_one.append(l)
        else:
            Synonym_Word = Synonym(sym_point, UniqueTokens())
            if Synonym_Word is not None:
                l = symptom_dict[Synonym_Word]
                diseases_one.append(l)

    if not diseases_one:
        diseases = []
    try:
        common_diseases = set(diseases_one[0])
    except:
        return 'Unknown Disease'
    for disease in diseases_one[1:]:
        common_diseases.intersection_update(disease)
    
    diseases = list(common_diseases)
    
    if len(diseases) == 0:
        append_disease = []
        for disease in diseases_one:
            append_disease  = append_disease + disease
        diseases = append_disease



    # Using Random Forest
    clf = RandomForestClassifier(n_estimators=750)
    symptoms_embedding = np.mean(output.last_hidden_state.numpy()[0], axis=0).reshape(1, -1)
    if len(diseases) == 1:
        return diseases[0]
    elif len(diseases) > 1:
        clf.fit(np.vstack([np.mean(output.last_hidden_state.numpy()[0], axis=0).reshape(1, -1) for _ in range(len(diseases))]), diseases)
        predicted_disease = clf.predict(symptoms_embedding)
        return predicted_disease[0]
    else:
        return "Unknown Disease"
