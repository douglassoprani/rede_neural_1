from PIL import Image
import face_recognition

image = face_recognition.load_image_file("pessoal.jpg")

faces = face_recognition.face_locations(image)

print("Achamos {} rostos nesta foto.".format(len(faces)))

for lugar_faces in faces:
	top, right, bottom, left = lugar_faces
	print("A face esta localizada no pixel Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
	imagem_rosto = image[top:bottom, left:right]
	mostrar = Image.fromarray(imagem_rosto)
	mostrar.show()

	
