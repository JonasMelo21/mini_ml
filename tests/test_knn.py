from src.knn import KNearestNeighbors
from src.linear_algebra import Vector

def test_knn_obvious_groups():
    """
    Testing the following situatio/problem:
        - there is a class full of students
        - the students can be divided in two groups
        - the groups are: Near to the teacher (label 0) or away from the teacher (label 1)
        - the students have their posisitions measured as vectors 
        - students near the teacher are assigned to vectors with low absolute value
        - near the teacher students:
            - [[1,1],[1,2],[2,1]]
        - the students far to the teacher have their posisitions measured as high absolute value
        - far the teacher students:
            - [[10,10],[10,11],[11,10]]
        - let's try to assign a label to an obvious 'near-to-the-teacher-student' 
        - and an obvious 'far-away-from-the-teacher-student'
            - [1.5,1.5] -> it has to be assigned as 0 (near to the teacher)
            - [10.5,10.5] -> it has to be assigned as 1 (far from the teacher)

    """

    # Defining training data
    student_position = [[1,1],[1,2],[2,1],[10,10],[10,11],[11,10]]
    x_train = [Vector(item) for item in student_position]
    y_train = [0,0,0,1,1,1]

    # Instacing the model
    model = KNearestNeighbors(3)
    model.fit(x_train,y_train)


    # Pedicting the 'near-the-teacher-student'
    near_student = Vector([1.5,1.5])
    near_student_label = model.predict(near_student)

    # Predicting the 'far-from-the-teacher-student' label
    far_student = Vector([10.5,10.5])
    far_student_label = model.predict(far_student)

    # Checking model accuraccy
    assert near_student_label == 0
    assert far_student_label == 1