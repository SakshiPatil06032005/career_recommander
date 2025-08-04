from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and data
model = joblib.load('model.pkl')
career_df = pd.read_csv('career_data.csv')

@app.route('/')
def index():
    likes = sorted(career_df['Likes'].unique())
    hobbies = sorted(career_df['Hobby'].unique())
    return render_template('index.html', likes=likes, hobbies=hobbies)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        math = int(request.form['math'])
        bio = int(request.form['bio'])
        english = int(request.form['english'])
        likes = request.form['likes']
        hobby = request.form['hobby']

        input_df = pd.DataFrame([[math, bio, english, likes, hobby]],
                                columns=['Math', 'Biology', 'English', 'Likes', 'Hobby'])
        prediction = model.predict(input_df)[0]

        return render_template('result.html',
                               prediction=prediction,
                               math=math, bio=bio, english=english,
                               likes=likes, hobby=hobby)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)