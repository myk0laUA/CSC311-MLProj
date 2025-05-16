import pandas as pd
import re
import model  # Imports the exported model (model.py) from m2cgen

###############################################################################
# Configuration
###############################################################################

# This is the final list of columns in preprocessed data
TRAINING_COLUMNS = [
    "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
    "Q2: How many ingredients would you expect this food item to contain?",
    "Q4: How much would you expect to pay for one serving of this food item?",
    "Q8: How much hot sauce would you add to this food item?",
    # -- Q3(Setting) columns
    "Q3(Setting):At a party",
    "Q3(Setting):Late night snack",
    "Q3(Setting):Week day dinner",
    "Q3(Setting):Week day lunch",
    "Q3(Setting):Weekend dinner",
    "Q3(Setting):Weekend lunch",
    "Q3_Token_Count",
    # -- Q5(Movie) columns
    "Q5(Movie):john","Q5(Movie):finding","Q5(Movie):ninja","Q5(Movie):big","Q5(Movie):chance",
    "Q5(Movie):pizza","Q5(Movie):drift","Q5(Movie):2012","Q5(Movie):sushi","Q5(Movie):kung",
    "Q5(Movie):bill","Q5(Movie):man","Q5(Movie):story","Q5(Movie):your","Q5(Movie):anime",
    "Q5(Movie):turtles","Q5(Movie):name","Q5(Movie):godfather","Q5(Movie):dreams","Q5(Movie):inc",
    "Q5(Movie):kid","Q5(Movie):mutant","Q5(Movie):japanese","Q5(Movie):jiro","Q5(Movie):hour",
    "Q5(Movie):away","Q5(Movie):dictator","Q5(Movie):cars","Q5(Movie):avengers","Q5(Movie):cloudy",
    "Q5(Movie):wick","Q5(Movie):teenage","Q5(Movie):fu","Q5(Movie):spider","Q5(Movie):meatballs",
    "Q5(Movie):shawarma","Q5(Movie):monsters","Q5(Movie):rush","Q5(Movie):deadpool","Q5(Movie):kill",
    "Q5(Movie):panda","Q5(Movie):aladdin","Q5(Movie):alone","Q5(Movie):nemo","Q5(Movie):spiderman",
    "Q5(Movie):home","Q5(Movie):tokyo","Q5(Movie):spirited","Q5(Movie):madagascar","Q5_Token_Count",
    # -- Q6 (drink) one-hot columns
    "Alcohol","Bubble Tea","Coke","Energy Drink","Fanta","Ginger Ale","Hot Chocolate","Juice","Milk",
    "No Drink","Root Beer","Soup","Sparkling Water","Sprite","Tea","Unknown","Water",
    # -- Q7(Setting) columns
    "Q7(Setting):Friends","Q7(Setting):Parents","Q7(Setting):Siblings","Q7(Setting):Strangers","Q7(Setting):Teachers",
    "Q7_Token_Count"
]

# Your final model might map numeric predictions back to these classes:
CLASS_MAPPING = {
    "0": "Pizza",
    "1": "Shawarma",
    "2": "Sushi"
}

###############################################################################
# Preprocessing Functions
###############################################################################

def convert_to_number(value):
    """
    Converts a string to an integer if possible.
    - Tries int(value)
    - Finds a float in the string with regex
    - Returns None if no match
    """
    if not isinstance(value, str):
        return None

    value = value.strip().lower()
    # 1) Try direct integer
    try:
        return int(value)
    except ValueError:
        pass

    # 2) Look for float-like substring
    numbers = re.findall(r"\d+\.?\d*", value)
    if numbers:
        return int(round(float(numbers[0])))

    return None

def clean_drink(drink):
    """
    Maps free-text drink responses to standardized categories.
    Returns 'Unknown' if no match is found.
    """
    if not isinstance(drink, str) or not drink.strip():
        return "Unknown"

    d = drink.lower().strip()
    # Check for "no drink"
    if re.search(r"\b(no\s*drink|none|nothing)\b", d):
        return "No Drink"

    cat = "Unknown"
    mapping_order = [
        (r"\b(miso\s*soup|soup)\b", "Soup"),
        (r"\b(sprite|7up|sierra\s*mist)\b", "Sprite"),
        (r"\b(diet\s*coke|coke\s*zero|cola|pepsi|dr\.?\s*pepper|pop|soda|mountain\s*dew|jarritos|barbican|ramune|soft\s*drink)\b", "Coke"),
        (r"\b(orange\s+crush|crush|fanta)\b", "Fanta"),
        (r"\broot\s*beer\b", "Root Beer"),
        (r"\b(ginger\s*ale|canada\s*dry)\b", "Ginger Ale"),
        (r"\b(sparkling\s*water|san\s+pellegrino|perrier|club\s*soda|spindrift)\b", "Sparkling Water"),
        (r"\bhot\s+chocolate\b", "Hot Chocolate"),
        (r"\b(hot\s*water|mineral\s*water|water)\b", "Water"),
        (r"\b(milk|chocolate\s+milk|milkshake|ayran|yogurt|calpis|yakult|lassi|laban|leban|dairy)\b", "Milk"),
        (r"\b(energy\s*drink|red\s*bull|monster|gatorade|powerade)\b", "Energy Drink"),
        (r"\b(bubble\s*tea|milk\s*tea|boba)\b", "Bubble Tea"),
        (r"\b(beer|wine|rum|whiskey|gin|champagne|baijiu|sake|soju|vodka|kraken|cocktail)\b", "Alcohol"),
        (r"\b(iced\s*tea|ice\s*tea|green\s*tea|black\s*tea|earl\s*grey|matcha|oolong|kombucha|jasmine\s*tea|tea)\b", "Tea"),
        (r"\b(juice|punch|smoothie|lemonade|mango\s*pulp)\b", "Juice"),
    ]
    for pattern, category in mapping_order:
        if re.search(pattern, d):
            cat = category
            break
    return cat

def encode_drinks(df, col_name):
    """
    One-hot encodes the Q6 drink column.
    """
    df[col_name] = df[col_name].fillna("Unknown").astype(str)
    drink_onehots = df[col_name].str.get_dummies()
    df.drop(columns=[col_name], inplace=True)
    df = pd.concat([df, drink_onehots], axis=1)
    return df

def encode_reminds(df, col_name):
    """
    For Q7, str.get_dummies(sep=',') => one-hot columns + Q7_Token_Count.
    """
    df[col_name] = df[col_name].fillna("").astype(str)
    q7_onehots = df[col_name].str.get_dummies(sep=",")
    q7_onehots.columns = [f"Q7(Setting):{c}" for c in q7_onehots.columns]
    df.drop(columns=[col_name], inplace=True)
    df = pd.concat([df, q7_onehots], axis=1)
    df["Q7_Token_Count"] = q7_onehots.sum(axis=1)
    return df

def process_q3(df, col_name):
    """
    Splits Q3 by commas => one-hot columns => Q3_Token_Count.
    """
    if col_name not in df.columns:
        return df
    df[col_name] = df[col_name].fillna("").astype(str)
    q3_onehots = df[col_name].str.get_dummies(sep=",")
    q3_onehots.columns = [f"Q3(Setting):{x}" for x in q3_onehots.columns]
    df.drop(columns=[col_name], inplace=True)
    df = pd.concat([df, q3_onehots], axis=1)
    df["Q3_Token_Count"] = q3_onehots.sum(axis=1)
    return df

def process_q5(df, col_name):
    """
    One-hot encodes known movie tokens + Q5_Token_Count.
    """
    Vocabulary = [
        'turtles','teenage','mutant','sushi','jiro','away','pizza','spirited','your','shawarma',
        'aladdin','dictator','tokyo','japanese','bill','kill','drift','ninja','home','alone',
        'avengers','nemo','monsters','dreams','finding','name','spider','godfather','anime',
        'fu','kung','inc','2012','madagascar','chance','cloudy','meatballs','cars','john',
        'wick','spiderman','panda','man','hour','rush','deadpool','big','kid','story'
    ]
    vocab_set = set(v.lower() for v in Vocabulary)

    df[col_name] = df[col_name].fillna("").astype(str).str.lower()
    onehot_rows = []
    for val in df[col_name]:
        tokens = set(re.findall(r"\b\w+\b", val))
        row = [1 if word in tokens else 0 for word in vocab_set]
        onehot_rows.append(row)

    onehot_df = pd.DataFrame(onehot_rows, columns=[f"Q5(Movie):{w}" for w in vocab_set])
    onehot_df["Q5_Token_Count"] = onehot_df.sum(axis=1)
    df.drop(columns=[col_name], inplace=True)
    df = pd.concat([df, onehot_df], axis=1)
    return df

def process_q8(df, col_name):
    """
    Maps Q8 text to numeric levels:
      0 = None
      1 = "I will have some of this food item with my hot sauce"
      2 = "A little (mild)"
      3 = "A moderate amount (medium)"
      4 = "A lot (hot)"
    """
    if col_name in df.columns:
        q8_map = {
            "None": 0,
            "I will have some of this food item with my hot sauce": 1,
            "A little (mild)": 2,
            "A moderate amount (medium)": 3,
            "A lot (hot)": 4
        }
        df[col_name] = df[col_name].fillna("None").replace(q8_map)
    return df

def process_q1(df, col_name):
    """
    If needed, convert Q1 to numeric. Otherwise, do minimal processing.
    """
    if col_name in df.columns:
        df[col_name] = df[col_name].fillna("").astype(str)
        df[col_name] = df[col_name].apply(convert_to_number)
    return df

###############################################################################
# Main predict_all function
###############################################################################

def predict_all(csv_file):
    """
    1) Reads new data from csv_file
    2) Applies transformations from your final_script
    3) Reindexes columns to match the trained model
    4) Uses model.score(...) for predictions
    5) Writes 'predictions.csv' and returns the list of predicted labels
    """
    df = pd.read_csv(csv_file)

    # Column references 
    Q1 = "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)"
    Q2 = "Q2: How many ingredients would you expect this food item to contain?"
    Q3 = "Q3: In what setting would you expect this food to be served? Please check all that apply"
    Q4 = "Q4: How much would you expect to pay for one serving of this food item?"
    Q5 = "Q5: What movie do you think of when thinking of this food item?"
    Q6 = "Q6: What drink would you pair with this food item?"
    Q7 = "Q7: When you think about this food item, who does it remind you of?"
    Q8 = "Q8: How much hot sauce would you add to this food item?"

    # Preprocess each column
    df = process_q1(df, Q1)

    if Q2 in df.columns:
        df[Q2] = df[Q2].astype(str).apply(convert_to_number)

    df = process_q3(df, Q3)

    if Q4 in df.columns:
        df[Q4] = df[Q4].astype(str).apply(convert_to_number)

    if Q5 in df.columns:
        df = process_q5(df, Q5)

    if Q6 in df.columns:
        df[Q6] = df[Q6].astype(str).apply(clean_drink)
        df = encode_drinks(df, Q6)

    if Q7 in df.columns:
        df = encode_reminds(df, Q7)

    df = process_q8(df, Q8)

    # Use median to handle NaNs
    df[Q2] = df[Q2].fillna(-1)     # Q2 median: 5.0
    df[Q4] = df[Q4].fillna(-1)    # Q4 median: 10.0
    df[Q8] = df[Q8].fillna(-1)     # Q8 median: 3.0

    # Drop columns not used (like 'Label' or 'id')
    if "Label" in df.columns:
        df.drop(columns=["Label"], inplace=True)
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # Reindex to match EXACT training columns. Missing => 0, extra => dropped.
    df = df.reindex(columns=TRAINING_COLUMNS, fill_value=0)

    # Make predictions row by row using the m2cgen-exported model
    predictions = []
    for _, row in df.iterrows():
        # Convert row to float list
        x = [float(v) for v in row.tolist()]
        # model.score(x) might return a single index or a probability vector
        pred = model.score(x)

        # If 'pred' is a list (probabilities), pick the argmax
        if hasattr(pred, "__iter__") and not isinstance(pred, str):
            pred_list = list(pred)
            if len(pred_list) > 1:
                pred_index = max(range(len(pred_list)), key=lambda i: pred_list[i])
                pred = pred_index

        # Convert numeric index -> class label
        label_str = CLASS_MAPPING.get(str(pred), str(pred))
        predictions.append(label_str)

    # Save predictions to CSV and return them
    out_df = pd.DataFrame(predictions, columns=["Prediction"])
    out_df.to_csv("predictions.csv", index=False)
    return predictions
