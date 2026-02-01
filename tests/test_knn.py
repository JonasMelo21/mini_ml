from src.knn import KNNClassifier
from src.linear_algebra import Vector

def test_knn_obvious_groups():
    """
    Test scenario:
        - There is a classroom of students
        - Students are divided into two groups:
            - Near the teacher (label 0)
            - Far from the teacher (label 1)
        - Positions are measured as vectors
        - Near the teacher: [[1,1],[1,2],[2,1]]
        - Far from the teacher: [[10,10],[10,11],[11,10]]
        - Predict labels for obvious cases:
            - [1.5,1.5] -> should be 0 (near)
            - [10.5,10.5] -> should be 1 (far)
    """
    # Training data
    student_position = [[1,1],[1,2],[2,1],[10,10],[10,11],[11,10]]
    x_train = [Vector(item) for item in student_position]
    y_train = [0,0,0,1,1,1]
    # Instantiate model
    model = KNNClassifier(3)
    model.fit(x_train, y_train)
    # Predict near student
    near_student = Vector([1.5,1.5])
    near_student_label = model.predict(near_student)
    # Predict far student
    far_student = Vector([10.5,10.5])
    far_student_label = model.predict(far_student)
    # Check accuracy
    assert near_student_label == 0
    assert far_student_label == 1