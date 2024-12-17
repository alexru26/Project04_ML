import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
my_drawing_specs = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
mp_face_mesh = mp.solutions.face_mesh

def draw_landmark(image, face_landmarks):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style()
    )

    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=my_drawing_specs
    )

    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=my_drawing_specs
    )

def get_landmark(bluff, num):
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
    ) as face_mesh:
        time.sleep(0.1)
        success, image = cap.read()

        results = face_mesh.process(image)

        if results.multi_face_landmarks is None:
            print("No faces found")
            return

        dir = "bluff/" if bluff else "nonbluff/"
        name = "bluff" + str(num) if bluff else "nonbluff" + str(num)

        crop_img = image[104:616, 384:896]

        cv2.imwrite("../data/no_landmark/"+dir+name+'.png', cv2.flip(crop_img, 1))

        landmarks = results.multi_face_landmarks[0]
        draw_landmark(image, landmarks)

        cv2.imwrite("../data/landmark/"+dir+name+'.png', cv2.flip(crop_img, 1))

    cap.release()
    cv2.destroyAllWindows()