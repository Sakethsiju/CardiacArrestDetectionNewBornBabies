from django.shortcuts import render,redirect
from django.contrib import  messages

from mainapp.models import *
from adminapp.models import *
from userapp.models import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE

# Create your views here.

def admin_dashboard(request):
    pending_user_count = UserDetails.objects.filter(user_status="pending").count()
    all_users_count = UserDetails.objects.exclude(user_status="pending").count()
    accepted_users_count=UserDetails.objects.filter(user_status="accepted").count()
    rejected_users_count=UserDetails.objects.filter(user_status="rejected").count()

    context = {
        'pending_users': pending_user_count,
        'all_users':all_users_count,
        'accepted_users': accepted_users_count,
        'rejected_users': rejected_users_count
    }
    return render(request, 'admin/admin-dashboard.html', context)

def pending_users(request):
    pending_users = UserDetails.objects.filter(user_status="pending")
    context = {
        'pending_users': pending_users
    }
    return render(request, 'admin/pending-users.html', context)

def deleteAll_penging(request):
    UserDetails.objects.filter(user_status="pending").delete()
    messages.success(request, 'All pending users deleted successfully!')
    return redirect('pending_users')


def all_users(request):
    all_users = UserDetails.objects.exclude(user_status="pending")
    context = {
        'all_users': all_users
    }
    return render(request, 'admin/all-users.html', context)

def deleteAll_users(request):
    UserDetails.objects.exclude(user_status="pending").delete()
    messages.success(request, 'All users deleted successfully!')
    return redirect('all_users')

def accept_user(request, id):
    dbUser = UserDetails.objects.get(user_id = id)
    dbUser.user_status = "accepted"
    dbUser.save()
    messages.success(request, 'User Accepted Successfully')
    return redirect('pending_users')

def reject_user(request,id):
    dbUser = UserDetails.objects.get(user_id = id)
    dbUser.user_status = "rejected"
    dbUser.save()
    messages.success(request, 'User Rejected Successfully')
    return redirect('pending_users')

def change_status(request, id):
    dbUser = UserDetails.objects.get(user_id=id)
    if  dbUser.user_status == "accepted":
        dbUser.user_status =  "rejected"
    elif  dbUser.user_status == "rejected":
        dbUser.user_status = "accepted"
    dbUser.save()
    messages.success(request, 'User Status Changed Successfully')
    return redirect('all_users')

def delete_user(request,id):
    dbUser = UserDetails.objects.get(user_id=id)
    dbUser.delete()
    messages.success(request, 'User Deleted Successfully')
    return redirect('all_users')

def new_leads(request):
    new_leads = UserContacts.objects.all()
    context = {
        'new_leads': new_leads
    }
    return render(request, 'admin/new-leads.html', context)

def deleteLead(request,id):
    dbLead = UserContacts.objects.get(user_id=id)
    dbLead.delete()
    messages.success(request, 'Lead Deleted Successfully')
    return redirect('new_leads')

def deleteAll_leads(request):
    UserContacts.objects.all().delete()
    messages.success(request, 'All Leads Deleted Successfully')
    return redirect("new_leads")


def upload_dataset(request):
    if request.method == 'POST':
        file = request.FILES['dataset_file']
        file_size = str((file.size)/1024) +' kb'
        formated_file_size = str(int(file.size/1024)) + ' kb'
        Datasets_Details.objects.create(file_size = file_size, dataset_name = file, formated_file_size = formated_file_size)
        messages.success(request, 'Dataset Uploaded Successfully')
    return render(request, 'admin/upload-dataset.html')

def view_datasets_list(request):
    datasets_list = Datasets_Details.objects.all()
    context = {
        'datasets_list': datasets_list
    }
    return render(request, 'admin/view-dataset-list.html', context)

def delete_dataset(request, id):
    dbDataset = Datasets_Details.objects.get(dataset_id=id)
    dbDataset.delete()
    messages.success(request, 'Dataset Deleted Successfully')
    return redirect('view_datasets_list')

def view_dataset_result(request, id):
    dbDataset = Datasets_Details.objects.get(dataset_id=id)
    file = str(dbDataset.dataset_name)
    df = pd.read_csv(f'./media/{file}')
    table = df.to_html(table_id='data_table')
    return render(request, 'admin/view-dataset-result.html', {'t':table})


def train_and_test(request):
    return 

def admin_feedbacks(request):
    feedbacks = Feedback.objects.all()
    context = {
        "feedbacks":  feedbacks
    }
    return render(request, 'admin/feedbacks.html', context)

def deleteFeed(request,id):
    dbFeed = Feedback.objects.get(Feed_id=id)
    dbFeed.delete()
    messages.success(request,'Feedback Deleted Successfully')
    return redirect('admin_feedbacks')

def deleteSentFeed(request,id):
    dbFeed = Feedback.objects.get(Feed_id=id)
    dbFeed.delete()
    messages.success(request,'Feedback Deleted Successfully')
    return redirect('sentiment_analysis')

def deleteAllFeeds(request):
    Feedback.objects.all().delete()
    messages.success(request, 'All Items all deleted')
    return redirect('admin_feedbacks')

def deleteAllsentFeeds(request):
    Feedback.objects.all().delete()
    messages.success(request, 'All Items all deleted')
    return redirect('sentiment_analysis')

def sentiment_analysis(request):
    feedbacks = Feedback.objects.all()
    context = {
        "feedbacks":  feedbacks
    }
    return render(request, 'admin/sentiment-analysis.html', context)

def feedbacks_graph(request):
    vp_count = Feedback.objects.filter(Sentiment = "very positive").count()
    p_count = Feedback.objects.filter(Sentiment = "positive").count()
    vn_count = Feedback.objects.filter(Sentiment = "very negative").count()
    neg_count = Feedback.objects.filter(Sentiment = "negative").count()
    n_count = Feedback.objects.filter(Sentiment = "neutral").count()
    context = {
        "vp_count": vp_count,
        "p_count": p_count,
        "vn_count": vn_count,
        "neg_count": neg_count,
        "n_count":  n_count
    }
    return render(request, 'admin/feedbacks-graph.html', context)
# data exploration

def create_proportion_chart(df, ax):
    if 'sex' in df.columns:
        sex_target_proportion = df.groupby('sex')['target'].value_counts(normalize=True).unstack().fillna(0)
        # Plot the proportions
        sex_target_proportion.plot(kind='bar', stacked=True, color=['lightblue', 'salmon'])
    else:
        ax.set_title('Column "sex" not found')

def create_boxplot(df, ax):
    if 'User_Age' in df.columns:
        sns.boxplot(df['User_Age'].dropna(), color='green', ax=ax)
        ax.set_title('Boxplot of User_Age')
        ax.set_xlabel('User_Age')
    else:
        ax.set_title('Column "User_Age" not found')

def create_lineplot(df, ax):
    if 'Account_Age' in df.columns and 'User_Age' in df.columns:
        sns.lineplot(x='Account_Age', y='User_Age', data=df, ax=ax)
        ax.set_title('Line Plot of Account_Age vs. User_Age')
        ax.set_xlabel('Account_Age')
        ax.set_ylabel('User_Age')
    else:
        ax.set_title('Required columns "Account_Age" or "User_Age" not found')

def create_piechart(df, ax):
    if 'Status' in df.columns:
        ax.pie(df["Status"].value_counts().values, labels=df["Status"].value_counts().index, autopct="%.02f%%")
        ax.set_title("Status Account Prediction")
    else:
        ax.set_title('Column "Status" not found')

def plot_to_base64(fig):
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close(fig) 
    return f'data:image/png;base64,{img_data}'

def data_exploration(request):
    if not Datasets_Details.objects.exists():
        messages.error(request, 'Upload Dataset First.')
        return render(request, 'admin/Upload-dataset.html', {})
    
    # Retrieve the latest uploaded dataset
    dataset = Datasets_Details.objects.last()
    try:
        df = pd.read_csv(dataset.dataset_name.path)
    except Exception as e:
        messages.error(request, f'Error reading dataset: {e}')
        return redirect('admin_dashboard')

    # Create subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Plot the Histogram
    create_proportion_chart(df, axes[0, 0])

    # Plot the Boxplot
    create_boxplot(df, axes[0, 1])

    # Plot the Line Plot
    create_lineplot(df, axes[1, 0])

    # Plot the Pie Chart
    create_piechart(df, axes[1, 1])

    # Convert the entire figure to base64-encoded image for rendering in HTML
    figure_img = plot_to_base64(fig)

    # Close the figure to free up resources
    plt.close(fig)

    context = {
        'figure_img': figure_img,
        'dataset': df.to_html(),
    }

    # messages.success(request, 'Data Exploration Analysis Completed Successfully')
    return render(request, 'admin/data-exploration.html', context)


def decision_tree(request):
    return render(request, 'admin/algorithms/decision-tree.html')

def decision_tree_result(request):
    data = Datasets_Details.objects.last()
    if data is None:
        messages.error(request, 'No dataset available')
        return redirect('decision_tree')
    
    file = str(data.dataset_name)
    df = pd.read_csv(f'./media/{file}')
    
    # Assume that the last column is the target variable
    X = df.drop('target', axis=1)
    y = df['target']
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,            # Features
    y_resampled,            # Target labels
    test_size=0.3, # Proportion of the dataset to include in the test split
    random_state=11 # Seed for the random number generator
    )
    
    # Initializing
    DT = DecisionTreeClassifier()
    DT.fit(X_train, y_train)
    from sklearn.model_selection import cross_val_score

    # y_predict = DT.predict(X_test)
    print('*'*20)

    # prediction
    train_pred=DT.predict(X_train)
    test_pred= DT.predict(X_test)
    print('*'*20)
    # accuracy
    print('Train accuracy:' , accuracy_score(y_train,train_pred))
    print('Test accuracy:' , accuracy_score(y_test,test_pred))

    print('*'*20)
    # cross validation   
    score= cross_val_score(DT,X,y,cv=5)
    print(score)
    print(score.mean())

    print('*'*20)
    #  prediction Summary by species
    print(classification_report(y_test, test_pred))

    print('*'*20)
    # Accuracy score
    DT_SC = accuracy_score(y_train,train_pred)
    print(f"{round(DT_SC*100,2)}% Accurate")

    print('*'*20)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, train_pred)*100
    test_accuracy = accuracy_score(y_test, test_pred)*100
    precision = precision_score(y_test, test_pred, average='weighted')
    recall = recall_score(y_test, test_pred, average='weighted')
    f1 = f1_score(y_test, test_pred, average='weighted')
    class_report = classification_report(y_test, test_pred)

        # Save results to the database
    name = "Decision Tree Algorithm"

    Decision_Tree_Algo.objects.create(
        Accuracy=train_accuracy,
        Precession=precision,
        F1_Score=f1,
        Recall=recall,
        Name=name
    )

    # Retrieve the latest GRADIENT_ALGO entry
    latest_algo = Decision_Tree_Algo.objects.last()
    messages.success(request, 'Algorithm executed successfully')

    return render(request, 'admin/algorithms/decision-tree-result.html',{"results": latest_algo})


def logistic_reg(request):
    return render(request, 'admin/algorithms/logistic-reg.html')

def logistic_reg_result(request):
    data = Datasets_Details.objects.last()
    if data is None:
        messages.error(request, 'No dataset available.')
        return redirect('logistic_reg')

    file = str(data.dataset_name)
    df = pd.read_csv(f'./media/{file}')

    # Assume that the last column is the target variable
    X = df.drop('target', axis=1)
    y = df['target']
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,            # Features
    y_resampled,            # Target labels
    test_size=0.3, # Proportion of the dataset to include in the test split
    random_state=11 # Seed for the random number generator
    )
    
    # Initializing 
    LR = LogisticRegression()
    LR.fit(X_train, y_train)


    # prediction
    train_prediction= LR.predict(X_train)
    test_prediction= LR.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score
    print('test accuracy:',accuracy_score(y_test,test_prediction))
    print('train accuracy:',accuracy_score(y_train,train_prediction))
    print('*'*20)

    # cross validation score
    from sklearn.model_selection import cross_val_score
    score=cross_val_score(LR,X,y,cv=5)
    print(score.mean())
    print('*'*20)

    print(classification_report(y_test,test_prediction))

    print('*'*20)


    lr_HSC = accuracy_score(y_test,test_prediction)
    print(f"{round(lr_HSC*100,2)}% Accurate")

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, train_prediction)*100
    test_accuracy = accuracy_score(y_test, test_prediction)*100
    precision = precision_score(y_test, test_prediction, average='weighted')
    recall = recall_score(y_test, test_prediction, average='weighted')
    f1 = f1_score(y_test, test_prediction, average='weighted')
    class_report = classification_report(y_test, test_prediction)

    # Prepare results for rendering
    # Save results to the database
    name = "Logistic Regression Algorithm"
    Logistic_Regression.objects.create(
        Accuracy=train_accuracy,
        Precession=precision,
        F1_Score=f1,
        Recall=recall,
        Name=name
    )

    # Retrieve the latest GRADIENT_ALGO entry
    latest_algo = Logistic_Regression.objects.last()
    messages.success(request, 'Algorithm executed successfully')
    return render(request, 'admin/algorithms/logistic-reg_result.html',{"results": latest_algo})


def SVM_algorithm(req):
    return render(req, 'admin/algorithms/svm-algo.html')


def SVM_Algo_result(request):
    data = Datasets_Details.objects.last()
    if data is None:
        messages.error(request, 'No dataset available.')
        return redirect('LGBM_algorithm')

    file = str(data.dataset_name)
    df = pd.read_csv(f'./media/{file}')

    # Assume that the last column is the target variable
    X = df.drop('target', axis=1)
    y = df['target']
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,            # Features
    y_resampled,            # Target labels
    test_size=0.3, # Proportion of the dataset to include in the test split
    random_state=11 # Seed for the random number generator
    )
    
    # Standardize the features
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both the train and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Support Vector Machine (SVM)
    svm = SVC()

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'kernel': ['linear', 'rbf'],  # Kernel types
        'gamma': ['scale', 'auto'],  # Kernel coefficient
    }

    grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    # Best model after Grid Search
    best_svm = grid_search.best_estimator_

    # Make predictions using the best model
    train_prediction = best_svm.predict(X_train_scaled)
    test_prediction = best_svm.predict(X_test_scaled)

    # Print evaluation metrics
    print('*' * 20)
    print('Test accuracy:', accuracy_score(y_test, test_prediction))
    print('Train accuracy:', accuracy_score(y_train, train_prediction))
    print('*' * 20)

    # Cross-validation score
    cross_val_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=5)
    print('Cross-validation score:', cross_val_scores.mean())
    print('*' * 20)

    # Classification report
    print(classification_report(y_test, test_prediction))

    print('*' * 20)

    # Final accuracy score
    svm_accuracy = accuracy_score(y_train, train_prediction)
    print(f"Model Accuracy: {round(svm_accuracy * 100, 2)}%")

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, train_prediction)*100
    test_accuracy = accuracy_score(y_test, test_prediction)*100
    precision = precision_score(y_test, test_prediction, average='weighted')
    recall = recall_score(y_test, test_prediction, average='weighted')
    f1 = f1_score(y_test, test_prediction, average='weighted')
    class_report = classification_report(y_test, test_prediction)

    # Prepare results for rendering
    # Save results to the database
    name = "SVM Algorithm"
    SVM_algo.objects.create(
        Accuracy=train_accuracy,
        Precession=precision,
        F1_Score=f1,
        Recall=recall,
        Name=name
    )

    # Retrieve the latest GRADIENT_ALGO entry
    latest_algo = SVM_algo.objects.last()
    messages.success(request, 'Algorithm executed successfully')
    return render(request, 'admin/algorithms/svm-algo-result.html', {"results": latest_algo})



def algorithms_comparision_graph(request):
    logistic_details = Logistic_Regression.objects.last()
    svm_details = SVM_algo.objects.last()
    decision_details = Decision_Tree_Algo.objects.last()  

    # Check if any model details are None
    if not all([decision_details, svm_details, logistic_details]):
        messages.error(request, 'Run the Algorithms First.')
          # Redirect to a different page, such as the home page or another view

    context = {
        'decision_accuracy': float(decision_details.Accuracy) if decision_details else 0,
        'svm_accuracy': float(svm_details.Accuracy) if svm_details else 0,
        'logistic_accuracy': float(logistic_details.Accuracy) if logistic_details else 0,
    }

    return render(request, 'admin/algorithms-comparision-graph.html', context)


def adminlogout(req):
    messages.success(req, "You are logged out.")
    return redirect("admin_login")

