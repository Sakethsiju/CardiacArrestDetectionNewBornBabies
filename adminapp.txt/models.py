from django.db import models

# Create your models here.

class Datasets_Details(models.Model):
    dataset_id = models.AutoField(primary_key = True)
    dataset_name = models.FileField(null=True)
    file_size = models.CharField(max_length = 100) 
    formated_file_size = models.CharField(max_length= 100, null=True)
    date_time = models.DateTimeField(auto_now = True)
    
    class Meta:
        db_table = 'datasets_details'

class Decision_Tree_Algo(models.Model):
    S_No = models.AutoField(primary_key=True)
    Accuracy = models.TextField(max_length=100)
    Precession = models.TextField(max_length=100)
    F1_Score = models.TextField(max_length=100)
    Recall = models.TextField(max_length=100)
    Name = models.TextField(max_length=100)

    class Meta:
        db_table = 'decision_tree_algo'


class Logistic_Regression(models.Model):
    S_No = models.AutoField(primary_key=True)
    Accuracy = models.TextField(max_length=100)
    Precession = models.TextField(max_length=100)
    F1_Score = models.TextField(max_length=100)
    Recall = models.TextField(max_length=100)
    Name = models.TextField(max_length=100)

    class Meta:
        db_table = 'logistic_regression_algo'



class SVM_algo(models.Model):
    S_No = models.AutoField(primary_key=True)
    Accuracy = models.TextField(max_length=100)
    Precession = models.TextField(max_length=100)
    F1_Score = models.TextField(max_length=100)
    Recall = models.TextField(max_length=100)
    Name = models.TextField(max_length=100)

    class Meta:
        db_table = 'SVM_algo'





