from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file("douglas2.jpg")

landmarks_face = face_recognition.face_landmarks(image)

print("Achei {} faces nesta foto.".format(len(landmarks_face)))

desenhar = Image.fromarray(image)
d = ImageDraw.Draw(desenhar)

for face in landmarks_face:

	for caract_face in face.keys():
		d.line(face[caract_face], width = 5)

desenhar.show()



