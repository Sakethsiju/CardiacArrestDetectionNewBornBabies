from django.shortcuts import render,redirect
from django.contrib import messages

from mainapp.models import *
from userapp.models import *
from adminapp.models import *

import pickle
import pandas as pd
import sklearn
import random

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Create your views here.
def user_dashboard(request):
    user_id = request.session["user_id"]
    dbUser = UserDetails.objects.get(user_id = user_id)
    context = {
        "user": dbUser,
    }
    return render(request, 'user/user-dashboard.html',context)

def detection(request):
    if request.method == 'POST':
        try:
            age = float(request.POST.get('age'))
            sex = float(request.POST.get('sex'))
            cp = float(request.POST.get('cp'))
            trestbps = float(request.POST.get('trestbps'))
            chol = float(request.POST.get('chol'))
            fbs = float(request.POST.get('fbs'))
            restecg = float(request.POST.get('restecg'))
            thalach = float(request.POST.get('thalach', ""))
            exang = float(request.POST.get('exang'))
            oldpeak = float(request.POST.get('oldpeak'))
            slope = float(request.POST.get('slope'))
            ca = float(request.POST.get('ca'))
            thal = float(request.POST.get('thal'))


        except (ValueError, TypeError):
            messages.warning(request, "Please enter valid numbers.")
            return redirect('detection')

        # Loading the saved model
        file_path = 'DT-cyber.pkl'  # Update with your model file path
        try:
            with open(file_path, 'rb') as file:
                loaded_model = pickle.load(file)

            # Validate the model type
            if not isinstance(loaded_model, sklearn.base.BaseEstimator):
                messages.error(request, "Loaded model is not compatible.")
                return redirect("detection")  # Redirect back to the form page

        except FileNotFoundError:
            messages.error(request, "Model file not found.")
            return redirect("detection")  # Redirect back to the form page
        except AttributeError as e:
            messages.error(request, f"Model loading error: {str(e)}")
            return redirect("detection")  # Redirect back to the form page

        prediction = loaded_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        prediction_result = int(prediction[0])
        request.session['prediction_result'] = prediction_result

        print(f"prediction_result: {prediction_result}")
        messages.success(request, 'Detection Successfull')
        return redirect("detection_result")  # Replace with your actual redirect URL
    return render(request, 'user/detection-dashboard.html')

def normalize_username(username):
    if not username:
        return 0  # Return 0 for empty usernames

    # Count the number of numerical characters
    numerical_count = sum(c.isdigit() for c in username)
    
    # Calculate the ratio
    ratio = numerical_count / len(username)
    
    return ratio


def normalize_fullname(fullname):
    # Remove spaces and calculate the length of the name
    stripped_name = fullname.replace(" ", "")
    length = len(stripped_name)

    if length == 0:
        return 0  # Return 0 for empty names after stripping spaces

    # Count the number of numerical characters
    numerical_count = sum(c.isdigit() for c in stripped_name)

    # Calculate the ratio
    ratio = numerical_count / length
    
    return ratio

# def normalize_description():
#     description_length = len("Dedicated professional with a passion for technology and knack for problem-solving")
#     print(description_length)
#     return description_length

def detection_instagram(request):
    if request.method == 'POST':
        try:
            # Convert to appropriate types
            # profile_pic_input = request.FILES.get('profile_pic', None)  # Use request.FILES for file uploads
            # if profile_pic_input == None:
            #     profile_pic = float(0)
            # else:
            #     profile_pic = float(1)

            profile_pic = float(request.POST.get('profile_pic', 0))

            username = request.POST.get('username_length', "")
            username_length = round(normalize_username(username), 2)

            fullname = request.POST.get('fullname', "")
            if str(fullname).lower() == "" or str(fullname).lower() == "na":
                fullname = ""
            fullname_words = float(len(str(fullname).split()))
            
            fullname_length = round(normalize_fullname(fullname), 2)

            name_equals_username_value = request.POST.get('name_equal_username', "")
            name_equals_username = float(name_equals_username_value)

            description = request.POST.get('description_length', "")
            if str(description).lower() == "" or str(description).lower() == "na":
                description = ""
            description_length = float(len(str(description)))

            # website_link = request.POST.get('external_url', "")
            # if str(website_link).lower() == "" or str(website_link).lower() == "na":
            #     external_url = float(0)
            # else:
            #     external_url = float(1)
            external_url = float(request.POST.get('external_url', 0))
            private = float(request.POST.get('private', 0))
            num_posts = float(request.POST.get('num_posts', 0))
            num_followers = float(request.POST.get('num_followers', 0))
            num_follows = float(request.POST.get('num_follows', 0))

            # Print values for debugging
            print(f"{profile_pic}, {username_length}, {fullname_words}, {fullname_length}, {name_equals_username}, {description_length}, {external_url}, {private}, {num_posts}, {num_followers}, {num_follows}")

            # Ensure correct column names and order
            column_names = ['profile_pic', 'username_length', 'fullname_words', 
                             'fullname_length', 'name_equals_username', 'description_length', 
                             'external_url', 'private', 'num_posts', 'num_followers', 'num_follows']
            
            input_data = pd.DataFrame([[profile_pic, username_length, fullname_words, fullname_length, 
                                        name_equals_username, description_length, external_url, 
                                        private, num_posts, num_followers, num_follows]], 
                                      columns=column_names)

            # Print DataFrame for debugging
            print(input_data)


            # normalize_description()
            # Loading the saved model
            file_path = 'xg_insta.pkl'  # Update with your model file path
            try:
                with open(file_path, 'rb') as file:
                    loaded_model = pickle.load(file)

                # Validate the model type
                if not isinstance(loaded_model, sklearn.base.BaseEstimator):
                    messages.error(request, "Loaded model is not compatible.")
                    return redirect("detection_instagram")

                # Make prediction
                prediction = loaded_model.predict(input_data)
                prediction_result = int(prediction[0])
                request.session['prediction_result'] = prediction_result

                messages.success(request, 'Detection Successful')
                return redirect("detection_result")
            
            except FileNotFoundError:
                messages.error(request, "Model file not found.")
                return redirect("detection_instagram")
            except AttributeError as e:
                messages.error(request, f"Model loading error: {str(e)}")
                return redirect("detection_instagram")
            
        except (ValueError, TypeError) as e:
            messages.warning(request, f"Please enter valid numbers: {str(e)}")
            return redirect('detection_instagram')
        except Exception as e:
            messages.error(request, f"An unexpected error occurred: {str(e)}")
            return redirect('detection_instagram')

    return render(request, 'user/detection-insta.html')

def detection_result(request):
    prediction_result = request.session.get('prediction_result', None)
    try:
        algo_result = Decision_Tree_Algo.objects.last()
    except:
        algo_result = None
    context = {
        'prediction_result': prediction_result,
        'algo_result': algo_result
    }
    return render(request, 'user/detection-result.html', context)

def user_profile(request):
    user_id = request.session["user_id"]
    print(user_id)
    dbUser = UserDetails.objects.get(user_id = user_id)

    if  request.method == "POST":
        
        user_name = request.POST.get('fullname')
        user_email = request.POST.get('email')
        user_password = request.POST.get('password')
        user_phone = request.POST.get('phone')
        user_age = request.POST.get('age')
        user_address = request.POST.get('address')

        if len(request.FILES) != 0:
            user_image = request.FILES['profileImg']
            dbUser.user_image = user_image
            dbUser.full_name = user_name
            dbUser.email = user_email
            dbUser.password = user_password
            dbUser.phone_number = user_phone
            dbUser.age = user_age
            dbUser.address = user_address

            dbUser.save()
            messages.success(request, "Profile Updated Successfully")
        else:
            dbUser.full_name = user_name
            dbUser.email = user_email
            dbUser.password = user_password
            dbUser.phone_number = user_phone
            dbUser.age = user_age
            dbUser.address = user_address

            dbUser.save()
            messages.success(request, "Profile Updated Successfully")


    context = {
        "user": dbUser
    }
    return render(request, 'user/profile.html', context)

def user_feedback(request):
    user_id = request.session["user_id"]
    dbUser = UserDetails.objects.get(user_id = user_id)
    if request.method == "POST":
        rating = request.POST.get('rating')
        comment = request.POST.get('comment')
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(comment)
        sentiment=None
        if score['compound']>0 and score['compound']<=0.5:
            sentiment='positive'
        elif score['compound']>=0.5:
            sentiment='very positive'
        elif score['compound']<-0.5:
            sentiment='negative'
        elif score['compound']<0 and score['compound']>=-0.5:
            sentiment='very negative'
        else :
            sentiment='neutral'
        Feedback.objects.create(Rating=rating,  Review=comment, Sentiment=sentiment, Reviewer=dbUser)
        messages.success(request, 'Feedback recorded successfully')
    return render(request, 'user/feedback.html')

def user_logout(req):
    req.session.flush()
    messages.success(req, 'You are logged out')
    return redirect("user_login")