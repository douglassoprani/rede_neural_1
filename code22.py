from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file("douglas2.jpg")

landmarks_face = face_recognition.face_landmarks(image)

desenhar = Image.fromarray(image)

for face in landmarks_face:
	d = ImageDraw.Draw(desenhar, 'RGBA')

	d.polygon(face['left_eyebrow'], fill=(68, 54, 39, 128))
	d.polygon(face['right_eyebrow'], fill=(68, 54, 39, 128))
	d.line(face['left_eyebrow'], fill=(68, 54, 39, 150), width = 6)
	d.line(face['right_eyebrow'], fill=(68, 54, 39, 150), width = 6)


	d.polygon(face['top_lip'], fill=(150, 0, 0, 128))
	d.polygon(face['bottom_lip'], fill=(150, 0, 0, 128))
	d.line(face['top_lip'], fill=(150, 0, 0, 64), width = 10)
	d.line(face['bottom_lip'], fill=(150, 0, 0, 64), width = 10)



	desenhar.show()









