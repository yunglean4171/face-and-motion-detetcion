import cv2
import datetime

def main():
	faceCascade = cv2.CascadeClassifier("/Users/janglin/Desktop/py/face-motion reg/face_recognition.xml")
	# define a video capture object
	video_capture = cv2.VideoCapture(0)
	#writing video
	frame_width = int(video_capture.get(3))
	frame_height = int(video_capture.get(4))
	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter(datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p") + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
	while(True):
		# Capture the video frame by frame
		ret, frame = video_capture.read()
		text="not detected"
		text1="not detected"
		timestamp = datetime.datetime.now()

		#face recognition
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30))

		if int(format(len(faces))) > 0: 
			#print("Found {0} faces!".format(len(faces)))
			text="detected"		
		else:
			text="not detected"

		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		#motion detection
		ret, frame1 = video_capture.read()	
		difference = cv2.absdiff(frame, frame1)  # find the difference between the frames
		gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (5, 5), 0)
		_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)  # create threshold
		dilated = cv2.dilate(thresh, None, iterations=3)
		contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for c in contours:
			if cv2.contourArea(c) < 5000:
				continue
			x, y, w, h = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
			text1="detected"

		# Display the resulting frame
		ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
		cv2.putText(frame, "Face status: {}".format(text), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.putText(frame, "Motion status: {}".format(text1), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
		
		#check if motion is detected if not change status text
		if text1 == "not detected":
			text1="detected"
		else:
			text1="not detected"

		out.write(frame)
		cv2.imshow('Press Q to quit', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# After the loop release the cap object
	video_capture.release()
	out.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()