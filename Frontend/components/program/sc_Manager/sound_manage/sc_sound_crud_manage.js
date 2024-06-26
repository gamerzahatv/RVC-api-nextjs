import React, { useState, useEffect } from "react";
import axios from 'axios';
import {Button} from "@react95/core";
import Swal from 'sweetalert2'

export function Sound_view() {
  const [data, setData] = useState({ count: 0, results: [] })
  const [currentPage, setcurrentPage] = useState(1)
  const itemsPerPage = 5
  const { count, results } = data


  useEffect(() => {
    console.log(currentPage)
    const fetchDataWithInterval = () => {
      if (currentPage <= 1) {
        setcurrentPage(1)
        fetchData(currentPage)
      }else{
        fetchData(currentPage)
      }
      
    }
  

  
    // Initial fetch
    fetchDataWithInterval();
  
    //  Set up interval
    const intervalId = setInterval(fetchDataWithInterval,2500); // Replace 5000 with your desired interval in milliseconds
  
    //  Clean up interval on component unmount or when currentPage changes
    return () => clearInterval(intervalId)
  
  }, [currentPage])
  

  const fetchData = async (start) => {
    try {
      const response = await axios.get(
        `${process.env.NEXT_PUBLIC_APP_URL}:${process.env.NEXT_PUBLIC_APP_Port}/manage-sound/view?start=${start}&limit=${itemsPerPage}`
      );
      setData(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const handleNextPage = () => {
    const start = currentPage + itemsPerPage
    setcurrentPage(start)
    fetchData(start)
  }

  const handlePreviousPage = () => {
    const start = currentPage - itemsPerPage
    setcurrentPage(start)
    fetchData(start)
  }


  const Deletefunc = (value,len) => {
    // console.log(len)
    axios.delete(`${process.env.NEXT_PUBLIC_APP_URL}:${process.env.NEXT_PUBLIC_APP_Port}/manage-sound/del?filename=${value}`)
    .then(response => {
      Swal.fire({
        title: 'Loading...',
        html: 'Please wait while we process your request.',
        showConfirmButton: false,
        allowOutsideClick: false,
        allowEscapeKey: false,
        allowEnterKey: false,
        didOpen: () => {
          // This function will be called when the modal is opened
          try{
            if ((len+4)%5 === 0){
              setcurrentPage(currentPage-5)
            }
            setTimeout(() => {
      
              Swal.close(); // Close the loading popup after your task is done
              
            }, 3000); // Adjust the timeout as needed
            
          }catch (error) {
            console.error(error)
          }
        },
      });
    })
    .catch(error => {
      console.error(error);
      fetchData(1)
    });
  
  }

  const Renamefunc = (oldfile) => {
    Swal.fire({
      title: "Submit your Renamefile",
      input: "text",
      inputAttributes: {
        autocapitalize: "off"
      },
      showCancelButton: true,
      confirmButtonText: "Look up",
      showLoaderOnConfirm: true,
      preConfirm: async (newfile) => {
        try {
          axios.put(`${process.env.NEXT_PUBLIC_APP_URL}:${process.env.NEXT_PUBLIC_APP_Port}/manage-sound/rename?oldfile=${oldfile}&newfile=${newfile}`)
          .then(response => {
            //console.log(response.data.status)
            if (response.data.status){
              Swal.fire({
                icon: "success",
                title: "Your work has been saved",
                showConfirmButton: false,
                timer: 1500
              });
            }else{
              Swal.fire({
                icon: "error",
                title: "Oops...",
                text: "Something went wrong!",
              });
            }
          });
        } catch (error) {
          Swal.showValidationMessage(`
            Request failed: ${error}
          `);
        }
      },
      allowOutsideClick: () => !Swal.isLoading()
    });
  };
  

  return (
    <div>
      <h1>Table CRUD SOUND</h1>
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
                <Button onClick={() => Renamefunc(item.text)} >Rename</Button>
                <Button onClick={() => Deletefunc(item.text,item.id)}>Delete</Button>
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
  );
}