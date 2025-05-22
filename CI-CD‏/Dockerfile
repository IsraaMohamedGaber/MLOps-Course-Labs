#use offical python image as a parent image
FROM python

# set the working directory in the container to /app
WORKDIR /app

# copy the content of the local src directory to the working directory
COPY requirements.txt . 
RUN pip install -r requirements.txt

# copy app coe
COPY . .

#Expose the port the app runs on
EXPOSE 8000

#Run the app using uvicorn  
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

