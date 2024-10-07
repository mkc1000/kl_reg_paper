from openai import OpenAI
import os
import glob
import re
import pickle
import random
import math

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def compare_teacher_behavior(transcript1, transcript2, quality="simpler", give_neither_option=True):
    options = '"Transcript 1" or "Transcript 2"'
    if give_neither_option:
        options += ' or "Neither"'

    prompt = f"""
    Analyze these two teacher-student conversation transcripts and determine in which one the teacher's behavior is {quality}. 
    Respond with ONLY {options}. Do not explain.

    Transcript 1:
    {transcript1}

    Transcript 2:
    {transcript2}

    Which transcript shows {quality} teacher behavior?
    """

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=10,
        temperature=1.0
    )

    result = response.choices[0].message.content.strip().lower()

    patterns = ['transcript 1', 'transcript 2']
    if give_neither_option:
        patterns.append('neither')
    
    for pattern in patterns:
        if pattern in result:
            return parse_result(pattern, give_neither_option)
    
    return None

def parse_result(result, give_neither_option):
    if result == "transcript 1":
        return 1
    elif result == "transcript 2":
        return 2
    elif result == "neither" and give_neither_option:
        return None
    else:
        return None

def get_filepaths(filepathstart):
    # Use glob to find all files that start with the given path
    matching_files = glob.glob(filepathstart + '*')
    # Convert to absolute paths
    absolute_paths = [os.path.abspath(f) for f in matching_files]
    return absolute_paths
    
def load_transcripts(filepathstart):
    filenames = get_filepaths(filepathstart)
    transcripts = []
    for filename in filenames:
        with open(filename, 'rb') as file:
            transcripts += pickle.load(file)
    return transcripts

def bernoulli_confidence_bound(p_hat, n):
    # 95% confidence interval
    z = 1.96
    return z / math.sqrt(n) * math.sqrt(p_hat * (1-p_hat))

def compare_models(model1='transcripts/budg_10', model2='transcripts/mixtral', quality="simpler", max_iter=256):
    transcripts1 = load_transcripts(model1)
    transcripts2 = load_transcripts(model2)
    random.shuffle(transcripts1)
    random.shuffle(transcripts2)
    count1, count2, total = 0, 0, 0
    for t1, t2, i in zip(transcripts1, transcripts2, range(max_iter)):
        result = compare_teacher_behavior(t1, t2, quality=quality, give_neither_option=False)
        total += 1
        if result == 1:
            count1 += 1
        if result == 2:
            count2 += 1
    frac1, frac2 = count1/total, count2/total
    print("----------------")
    print(quality)
    print(f"{model1.split('/')[-1]} vs. {model2.split('/')[-1]}")
    print(f"{frac1} +/- {bernoulli_confidence_bound(frac1, total)} vs. {frac2} +/- {bernoulli_confidence_bound(frac2, total)}")
    print(f"N = {total}")

if __name__ == '__main__':
    for quality in ["more complex/unpredictable", "better"]:
        compare_models(model1='transcripts/budg_10', model2='transcripts/mixtral', quality=quality, max_iter=300)
        compare_models(model1='transcripts/budg_20', model2='transcripts/mixtral', quality=quality, max_iter=300)
        compare_models(model1='transcripts/budg_20', model2='transcripts/budg_10', quality=quality, max_iter=300)