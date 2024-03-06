import axios from 'axios';
import {Button ,RadioButton,Input} from "@react95/core";
import React, { useState,useRef } from "react";



   

export function Upload_music(){
    const [selectedFilemusic, setSelectedFilemusic] = useState(null);
    const handleFileChangemusic = (e) => {
      const file = e.target.files[0]; // Get the first selected file
      if (file && file.size <= 70 * 1000 * 1000 && (file.name.endsWith('.mp3') || file.name.endsWith('.wav')) )  {
        setSelectedFilemusic(file)
  
        console.log(file)
      }else{
        e.target.value = null;
        alert('File support .wav and .mp3 File size  maximum limit of 70 MB.');
      }
    };
  
    const onfilesubmitRef = useRef(null);
    const handleSubmituploadmusic = async (e) => {
      e.preventDefault(); // Move this line to the beginning of the function
    
      if (!selectedFilemusic) {
        alert('Please select a valid file.');
        return;
      }
    
      // store the states in the form data
      const uploadmusicFormData = new FormData();
      uploadmusicFormData.append('audioFile', selectedFilemusic);
    
      try {
        const response = await axios({
          method: 'POST',
          url: 'http://192.168.1.38:5000/manage-sound/upload',
          data: uploadmusicFormData,
          headers: { 'Content-Type': 'multipart/form-data' },
        })
        if (response.status === 200) {
          console.log('Response:', response.data);
          //setSelectedFilemusic(null)
          onfilesubmitRef.current.value = null;
          alert('OK')
        } else {
          console.log('Unexpected status:', response.status);
        }
      } catch (error) {
        // handle error
        console.error(error);
      }
    };
  
    return (
      <>
      <form onSubmit={handleSubmituploadmusic}>
        <h3>Upload music</h3>
        <input type="file"  
          accept=".mp3,.wav"
          multiple={false}
          ref={onfilesubmitRef}
          onChange={handleFileChangemusic}
          >
        </input>
        <Button type="submit" > Upload</Button>
      </form>
      </>
    )
}
