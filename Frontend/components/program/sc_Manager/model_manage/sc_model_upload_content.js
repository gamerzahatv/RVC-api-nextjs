import axios from "axios";
import { Button, RadioButton, Input } from "@react95/core";
import React, { useState, useRef } from "react";
import Swal from "sweetalert2";
export function UploadModel() {
  const onfilesubmitRefpth = useRef(null);
  const onfilesubmitRefindex = useRef(null);

  const [selectedOptionModel, onChangeModel] = React.useState("have_index");
  const handleChangeModel = (e) => {
    onChangeModel(e.target.value);
  };

  //input name model
  const [inputValue, setInputValue] = useState("");
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  //pth file
  const [selectedFilepth, setSelectedFilepth] = useState(null)
  const handleFileChangepth = (e) => {
    const file = e.target.files[0]; // Get the first selected file
    if (file && file.size > 200 * 1024 * 1024) {
      alert("File size pth model exceeds the maximum limit of 200MB.");
      // You may want to clear the file input value here
      e.target.value = null;
    } else if (
      file &&
      file.size <= 200 * 1024 * 1024 &&
      file.name.endsWith(".pth")
    ) {
      setSelectedFilepth(file);
      //console.log('pth pass')
    } else {
      e.target.value = null;
      alert("Error");
    }
  };

  //indexfile 
  const [selectedFileindex, setSelectedFileindex] = useState(null);
  const handleFileChangeindex = (e) => {
    const file = e.target.files[0]; // Get the first selected file
    if (file && file.size > 200 * 1024 * 1024) {
      alert("File size index model exceeds the maximum limit of 200MB.");
      // You may want to clear the file input value here
      e.target.value = null;
    } else if (
      file &&
      file.size <= 200 * 1024 * 1024 &&
      file.name.endsWith(".index")
    ) {
      setSelectedFileindex(file);
      //console.log('index pass')
    } else {
      e.target.value = null;
      alert("Error");
    }
  };

  const handleClickupload = async (e) => {
    e.preventDefault();
  
    if (selectedOptionModel == null) {
      alert("Please input model name");
    } else if (selectedOptionModel === "have_index" && inputValue !== "") {
      const uploadmodelFormData = new FormData();  // Fix variable name here
      uploadmodelFormData.append("modelname", inputValue);
      uploadmodelFormData.append("pth", selectedFilepth);
      uploadmodelFormData.append("index", selectedFileindex);
      
  
      try {
        const response = await axios({
          method: "POST",
          url: "http://192.168.1.38:5000/manage-model/upload/index",
          data: uploadmodelFormData,  // Fix variable name here
          headers: { "Content-Type": "multipart/form-data" },
        });
  
        if (response.status === 200) {
          //console.log("Response:", response.data)
          setInputValue("");
          onfilesubmitRefindex.current.value = null
          onfilesubmitRefpth.current.value = null
          Swal.fire(response.data.Status)

        } else {
          console.log("Unexpected status:", response.status)
        }
      } catch (error) {
        console.error(error);
      }
    } else if (selectedOptionModel === "no_index" && inputValue !== "") {
      const uploadnoindexFormData = new FormData();
      uploadnoindexFormData.append("modelname", inputValue)
      uploadnoindexFormData.append("pth", selectedFilepth)
  
      try {
        const response = await axios({
          method: "POST",
          url: "http://192.168.1.38:5000/manage-model/upload/not-index",
          data: uploadnoindexFormData,
          headers: { "Content-Type": "multipart/form-data" },
        });
  
        if (response.status === 200) {
          //console.log("Response:", response.data.Status)
          Swal.fire(response.data.Status)
          setInputValue("");
          onfilesubmitRefpth.current.value = null
        }
      } catch (error) {
        console.error(error);
        Swal.fire(error);
        setInputValue("");
        onfilesubmitRefpth.current.value = null
      }
    } else {
      Swal.fire("Error Please input !");
    }
  };
  
  return (
    <>
    <form onSubmit={handleClickupload}>
      <h3>Upload model</h3>
      <p>Modelname</p>
      <Input
        type="text"
        value={inputValue}
        onChange={handleInputChange}
        placeholder="Type model name"
      />
      <br />
      <br />
      <RadioButton
        name="have_index"
        value="have_index"
        checked={selectedOptionModel === "have_index"}
        onChange={handleChangeModel}
      >
        have index file
      </RadioButton>
      <RadioButton
        name="no_index"
        value="no_index"
        checked={selectedOptionModel === "no_index"}
        onChange={handleChangeModel}
      >
        not have index file
      </RadioButton>
      <p>pth file</p>
      <input
        type="file"
        accept=".pth"
        multiple={false}
        ref={onfilesubmitRefpth}
        onChange={handleFileChangepth}
      />

      {selectedOptionModel === "have_index" && (
        <>
          <p>index file</p>
          <input
            type="file"
            accept=".index"
            multiple={false}
            ref={onfilesubmitRefindex}
            onChange={handleFileChangeindex}
          />
        </>
      )}
      <Button type="submit" > Upload model</Button>
    </form>
    </>
  );
}

export default UploadModel;
