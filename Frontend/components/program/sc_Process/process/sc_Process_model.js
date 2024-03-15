import React, { useState, useRef, useEffect  } from "react";
import Swal from "sweetalert2";
import axios from 'axios';

import {
  List,
  Fieldset,
  Dropdown,
  Button,
  Input,
  RadioButton,
  Range,
} from "@react95/core";

export function SC_Process_model() {
  const [selectsound, setselectsound] = useState("")
  const [selectmodel, setselectmodel] = useState("")
  const [selectpth, setselectpth] = useState("")
  const [selectindex, setselectindex] = useState("")

  // select pitchalgorithm
  const [selectedOptionpitchalgorithm, setSelectedOptionpitchalgorithm] =
    React.useState(null);
  const handleChangeOptionpitchalgorithm = (e) => {
    setSelectedOptionpitchalgorithm(e.target.value);
    //console.log(e.target.value);
  };

  //slider   median filtering
  const sliderRefmedianfiltering = useRef(null);
  const [sliderValuemedianfiltering, setSliderValuemedianfiltering] =
    useState(0);
  const handleSliderChangemedianfiltering = () => {
    // Access the current value of the slider using the ref
    const valuemedianfiltering = sliderRefmedianfiltering.current.value;

    // Update the state with the new value
    setSliderValuemedianfiltering(valuemedianfiltering);
    //console.log(valuemedianfiltering);
  };

  const handleInputChangemedianfiltering = (event) => {
    const value = event.target.value;
    setSliderValuemedianfiltering(value);
  };

  //slider & input index_rate
  const sliderRefvolume_index_rate = useRef(null);
  const [
    sliderValuevolume_index_rate,
    setSliderValuevolume_index_rate,
  ] = useState(0);
  const handleSliderChangevolume_index_rate = () => {
    // Access the current value of the slider using the ref
    const valuevolume_index_rate =
      sliderRefvolume_index_rate.current.value;

    // Update the state with the new value
    setSliderValuevolume_index_rate(valuevolume_index_rate);
    //console.log(valuevolume_index_rate)
  };

  const handleInputChangevolume_indexrate = (event) => {
    const valuevolume_index_rate = event.target.value;
    setSliderValuevolume_index_rate(valuevolume_index_rate);
  };

  //slider  protect voice
  const sliderRefprotect_voice = useRef(null);
  const [sliderValueprotect_voice, setSliderValueprotect_voice] = useState(0);
  const handleSliderChangeprotect_voice = () => {
    // Access the current value of the slider using the ref
    const valueprotect_voice = sliderRefprotect_voice.current.value;

    // Update the state with the new value
    setSliderValueprotect_voice(valueprotect_voice);
    //console.log(valueprotect_voice);
  };

  const handleInputChangeprotect_voice = (event) => {
    const valueprotect_voice = event.target.value;
    setSliderValueprotect_voice(valueprotect_voice);
  };


  //rms mix rate
  const sliderRefrms_mixrate = useRef(null);
  const [sliderValuerms_mixrate, setSliderValuerms_mixrate] = useState(0);
  const handleSliderChangerms_mixrate = () => {
    // Access the current value of the slider using the ref
    const valuerms_mixrate = sliderRefrms_mixrate.current.value;

    // Update the state with the new value
    setSliderValuerms_mixrate(valuerms_mixrate);
    //console.log(valueprotect_voice);
  };

  const handleInputChangerms_rate = (event) => {
    const valuerms_mixrate = event.target.value;
    setSliderValuerms_mixrate(valuerms_mixrate);
  };


  //resample
  const sliderRef_resample = useRef(null);
  const [sliderValue_resample, setSlider_resample] = useState(0);
  const handleSliderChange_resample = () => {
    // Access the current value of the slider using the ref
    const value_resample = sliderRef_resample.current.value;

    // Update the state with the new value
    setSlider_resample(value_resample);
    //console.log(valueprotect_voice);
  };

  const handleInput_resample = (event) => {
    const value_resample = event.target.value;
    setSlider_resample(value_resample);
  };

  const handleSubmit = async () => {
    try {
      const jsonData = {
        f0up_key:parseInt(inputValuef0up_key),
        input_path:`audio/${selectsound}`.toString(),
        index_path:`assets/weights/${selectindex}`.toString(),
        model_name:selectpth.toString(),
        f0method:selectedOptionpitchalgorithm.toString(),
        opt_path:`audio_output/${input_audiooutput}`.toString(),
        index_rate:parseFloat(sliderValuevolume_index_rate),
        filter_radius:parseInt(sliderValuemedianfiltering),
        resample_sr:parseInt(inputValue_samplerate),
        rms_mix_rate:parseFloat(sliderValuerms_mixrate),
        protect:parseFloat(sliderValueprotect_voice)
      }

      let timerInterval;
      Swal.fire({
        title: "Processing Model!",
        html: "I will close When process finish <b></b>  ",
        timerProgressBar: true,
        allowOutsideClick: false,
        didOpen: () => {
          Swal.showLoading();
          const timer = Swal.getPopup().querySelector("b");
          timerInterval = setInterval(() => {
            timer.textContent = `${Swal.getTimerLeft()}`;
          }, 100);
        },
        willClose: () => {
          clearInterval(timerInterval);
        }
      });
  
      const response = await axios.post(`${process.env.NEXT_PUBLIC_APP_URL}:${process.env.NEXT_PUBLIC_APP_Port}/infer`, jsonData, {
        headers: {
          'Content-Type': 'application/json'
        }
      })


      if (response.status === 200) {
        Swal.close(); // Close the Swal alert if the response is successful
        Swal.fire(`file output ${input_audiooutput}`);
      }
    } catch (error) {
      console.error('Error:', error)
    }
  }


  //f0 upkey
  const [inputValuef0up_key, setInputValuef0up_key] = useState(0);
  const handleInputChangef0up_key = (e) => {
    setInputValuef0up_key(e.target.value);
  };

  const handleInputChangepth = (e) => {
    setselectpth(e.target.value);
  };
  const handleInputChangeindex = (e) => {
    setselectindex(e.target.value);
  };
  
  //resample rate
  const [inputValue_samplerate, setInputValue_samplerate] = useState(0);
  const handleInputChange_samplerate = (e) => {
    setInputValue_samplerate(e.target.value);
  };

  //audio output after process
  const [input_audiooutput, setInputValue_audiooutput] = useState('sample.wav');
  const handleInput_audiooutput = (e) => {
    setInputValue_audiooutput(e.target.value);
  };


  return (
    <>
      <h2>Step 1 Choose a Model </h2>
      <SC_Process_selectmodel setselectmodel={setselectmodel} setselectpth={setselectpth} setselectindex={setselectindex}/>
      <div style={{ display: "flex", alignItems: "flex-start" }}>
        <Fieldset legend="Pth file" >
          <div>
          {/* <Input 
            value={selectpth} 
            onChange={handleInputChangepth}
          /> */}
          <p> Pth File : {selectpth} </p>
          
          </div>
        </Fieldset>
      </div>
      <div style={{ display: "flex", alignItems: "flex-start" }}>
          <Fieldset legend="Index file" >
            <div>
            {/* <Input 
              value={selectindex}
              onChange={handleInputChangeindex}
            /> */}
            <p> Index File : {selectindex} </p>
            
            </div>
          </Fieldset>
      </div>
          

      
      <h2>Step 2  Choose a Sound </h2>
      <p>It is recommended that the audio file extension be .wav or .mp3  input_path</p>
      {/* <YourComponent/> */}
      <SC_Process_selectsound setselectsound={setselectsound}/>
      <p>You Select : {selectsound}</p> 

      <h2>Step 3 Change pitch (Options) f0upkey</h2>
      <p>
        Transpose integer, number of semitones, raise by an octave: 12, lower by
        an octave: -12
      </p>
      <Input
        type="number" value={inputValuef0up_key} onChange={handleInputChangef0up_key}
      />
      <h2>Step 4 Select the pitch extraction algorithm  f0method</h2>
      <p>
      Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement
      </p>

      <Fieldset legend="Select the pitch extraction algorithm">
        <RadioButton
          name="working"
          value="pm"
          checked={selectedOptionpitchalgorithm === "pm"}
          onChange={handleChangeOptionpitchalgorithm}
        >
          pm
        </RadioButton>
        <RadioButton
          name="working"
          value="harvest"
          checked={selectedOptionpitchalgorithm === "harvest"}
          onChange={handleChangeOptionpitchalgorithm}
        >
          harvest
        </RadioButton>
        <RadioButton
          name="working"
          value="crepe"
          checked={selectedOptionpitchalgorithm === "crepe"}
          onChange={handleChangeOptionpitchalgorithm}
        >
          crepe
        </RadioButton>
        <RadioButton
          name="working"
          value="rmvpe"
          checked={selectedOptionpitchalgorithm === "rmvpe"}
          onChange={handleChangeOptionpitchalgorithm}
        >
          rmvpe
        </RadioButton>
      </Fieldset>
      <br></br>
      <h2>Step 5 median filtering (Options) </h2>
      <p>
      If ≥3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.
      </p>
      <Fieldset
        legend="Customize median filtering"
        className="contentmodelfield"
      >
        <Range
          min="0"
          max="7"
          value={sliderValuemedianfiltering}
          onChange={handleSliderChangemedianfiltering}
          ref={sliderRefmedianfiltering}
        />
        <Input
          className="generalrangecontent"
          value={sliderValuemedianfiltering}
          onChange={handleInputChangemedianfiltering}
        />
      </Fieldset>
      <p>median filtering Value: {sliderValuemedianfiltering}</p>

      <h2>Step 6 index_rate </h2>
      <p>Search feature ratio (controls accent strength, too high has artifacting)</p>
      <Fieldset
        legend="index rate"
        className="contentmodelfield"
      >
        <Range
          min="0"
          max="1"
          step="0.01"
          value={sliderValuevolume_index_rate}
          onChange={handleSliderChangevolume_index_rate}
          ref={sliderRefvolume_index_rate}
        />
        <Input
          className="generalrangecontent"
          value={sliderValuevolume_index_rate}
          onChange={handleInputChangevolume_indexrate}
        />
      </Fieldset>
      <p>
        Adjust the volume envelope scaling Value:{" "}
        {sliderValuevolume_index_rate}
      </p>

      <h2>Step 7 Protect voiceless consonants and breath sounds (Options)</h2>
      <p>
        Protect voiceless consonants and breath sounds to prevent artifacts such
        as tearing in electronic music. Set to 0.5 to disable. Decrease the
        value to increase protection, but it may reduce indexing accuracy:
      </p>
      <Fieldset
        legend="Protect voiceless consonants and breath sounds"
        className="contentmodelfield"
      >
        <Range
          min="0"
          max="0.5"
          step="0.01"
          value={sliderValueprotect_voice}
          onChange={handleSliderChangeprotect_voice}
          ref={sliderRefprotect_voice}
        />
        <Input
          className="generalrangecontent"
          value={sliderValueprotect_voice}
          onChange={handleInputChangeprotect_voice}
        />
      </Fieldset>
      <p>protect voice is : {sliderValueprotect_voice}</p>
      <h2>Step 8 Resample</h2>
      <p>Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling</p>
      <Fieldset
        legend="Resample"
        className="contentmodelfield"
      >
        <Range
          min="0"
          max="48000"
          step="1"
          value={sliderValue_resample}
          onChange={handleSliderChange_resample}
          ref={sliderRef_resample}
        />
        <Input
          className="generalrangecontent"
          value={sliderValue_resample}
          onChange={handleInputChange_samplerate}
        />
      </Fieldset>
      <h2>Step 9 RMS mixrate </h2>
      <p>Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume:</p>
      <Fieldset
        legend="Protect voiceless consonants and breath sounds"
        className="contentmodelfield"
      >
        <Range
          min="0"
          max="1"
          step="0.01"
          value={sliderValuerms_mixrate}
          onChange={handleSliderChangerms_mixrate}
          ref={sliderRefrms_mixrate}
        />
        <Input
          className="generalrangecontent"
          value={sliderValuerms_mixrate}
          onChange={handleInputChangerms_rate}
        />
      </Fieldset>
      
      <p>
          File output
      </p>
      <Input 
      value={input_audiooutput} onChange={handleInput_audiooutput}
      />

      <Button onClick={handleSubmit}>Convert</Button>
      
    </>
  );
}



export function SC_Process_selectmodel({ setselectmodel ,setselectpth ,setselectindex}) {
  const [data, setData] = useState({ count: 0, results: [] });
  const [currentPage, setcurrentPage] = useState(1);
  const itemsPerPage = 5;
  const { count, results } = data;

  useEffect(() => {
    //console.log(currentPage)
    const fetchDataWithInterval = () => {
      if (currentPage <= 1) {
        setcurrentPage(1);
        fetchData(currentPage);
      } else {
        fetchData(currentPage);
      }
    };

    // Initial fetch
    fetchDataWithInterval();

    // Set up interval
    const intervalId = setInterval(fetchDataWithInterval, 2500); // Replace 5000 with your desired interval in milliseconds

    // Clean up interval on component unmount or when currentPage changes
    return () => clearInterval(intervalId);
  }, [currentPage]);

  const fetchData = async (start) => {
    try {
      const response = await axios.get(
        `${process.env.NEXT_PUBLIC_APP_URL}:${process.env.NEXT_PUBLIC_APP_Port}/manage-model/view?start=${start}&limit=${itemsPerPage}`
      );
      setData(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const handleNextPage = () => {
    const start = currentPage + itemsPerPage;
    setcurrentPage(start);
    fetchData(start);
  };

  const handlePreviousPage = () => {
    const start = currentPage - itemsPerPage;
    setcurrentPage(start);
    fetchData(start);
  };

 
  const Selectfunc = (file, fileNames) => {
    //setselectmodel(file);
    //setselectfile(fileNames)
    //Swal.fire("You choose sound   ", file);
    //console.log(fileNames)
    //console.log("File Names:", fileNames);
    const pthFiles = fileNames.filter(fileName => fileName.endsWith('.pth'));
    const indexFiles = fileNames.filter(fileName => fileName.endsWith('.index'));
    
    // Set state with the selected file and both arrays
    setselectmodel(file);
    setselectpth(pthFiles); // Assuming you have separate states for .pth and .index files
    setselectindex(indexFiles);
    
    // Display message using SweetAlert
    Swal.fire("You chose sound", file);
    
    // Log both arrays to the console
    //console.log("Files ending with .pth:", pthFiles);
    //console.log("Files ending with .index:", indexFiles);
  };


  

  return (
    <>
    <div>
      <h2>Select MODEL</h2>
      <table >
        <thead>
          <tr>
            <th>ID</th>
            <th>MODEL</th>
            <th>Files</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {results.map((item, index) => (
            <tr key={index}>
              <td>{item.unique_number}</td>
              <td>{item.model_name}</td>
              <td>
              {item.files.map((file, fileIndex) => (
                <div key={fileIndex}>
                  {file.file_name}
                </div>
              ))}
              </td>
              <td>
              {/* <Button onClick={() => Selectfunc(item.model_name)}>Select</Button> */}
              <Button onClick={() => Selectfunc(item.model_name, item.files.map(file => file.file_name))}>Select</Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div >
        <button  onClick={handlePreviousPage} disabled={currentPage <= 1}>Previous</button>
        <button onClick={handleNextPage} disabled={currentPage + itemsPerPage > count}>Next</button>
      </div>
    </div>

    </>
  );
}


export function SC_Process_selectsound({ setselectsound }) {
  const [data, setData] = useState({ count: 0, results: [] });
  const [currentPage, setcurrentPage] = useState(1);
  const itemsPerPage = 5;
  const { count, results } = data;

  useEffect(() => {
    const fetchDataWithInterval = () => {
      if (currentPage <= 1) {
        setcurrentPage(1);
        fetchData(currentPage)
      } else {
        fetchData(currentPage)
      }
    }

    fetchDataWithInterval();
    const intervalId = setInterval(fetchDataWithInterval, 2500);
    return () => clearInterval(intervalId)

  }, [currentPage]);

  const fetchData = async (start) => {
    try {
      const response = await axios.get(`${process.env.NEXT_PUBLIC_APP_URL}:${process.env.NEXT_PUBLIC_APP_Port}/manage-sound/view?start=${start}&limit=${itemsPerPage}`);
      setData(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const handleNextPage = () => {
    const start = currentPage + itemsPerPage;
    setcurrentPage(start);
    fetchData(start);
  };

  const handlePreviousPage = () => {
    const start = currentPage - itemsPerPage;
    setcurrentPage(start);
    fetchData(start);
  };

  const Selectfunc = (file) => {
    setselectsound(file);
    Swal.fire("You choose sound   ", file);
  };

  return (
    <>
      <div style={{ display: "flex", alignItems: "flex-start" }}>
        <Fieldset legend="Choose a sound" className="contentmodelfield">
          <div>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Sound</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {results.map((item) => (
                  <tr key={item.id}>
                    <td>{item.id}</td>
                    <td>{item.text}</td>
                    <td>
                      <Button onClick={() => Selectfunc(item.text)}>Select</Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div>
              <button onClick={handlePreviousPage} disabled={currentPage <= 1}>Previous</button>
              <button onClick={handleNextPage} disabled={currentPage + itemsPerPage > count}>Next</button>
            </div>
          </div>
        </Fieldset>
      </div>
      <div>
      
      </div>
    </>
  );
}





