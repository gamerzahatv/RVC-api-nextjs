import React, { useState, useRef } from "react";
import {
  List,
  Fieldset,
  Dropdown,
  Button,
  Input,
  RadioButton,
  Range,
} from "@react95/core";
import { Awfxcg321303 } from "@react95/icons";
import * as S from "../../layoutStyling";
import { marginTop } from "@xstyled/styled-components";
import { SC_Process_model } from "./process/sc_Process_model";


function SC_Process_Layout({ closeSC_Process }) {
  return (
    <>
      <S.layoutMain
        // isMobile={isMobile}
        title={"SC_Process.exe"}
        closeModal={closeSC_Process}
        height="100%"
        icon={<Awfxcg321303 variant="32x32_4" />}
        menu={[
          {
            name: "Options",
            list: (
              <List>
                <List.Item onClick={closeSC_Process}>Close</List.Item>
              </List>
            ),
          },
        ]}
      >
        <S.layoutMainContent bg="white" boxShadow="in">
          <S.textModal>
          <SC_Process_model/>
          </S.textModal>
        </S.layoutMainContent>
      </S.layoutMain>
    </>
  );
}

export default  SC_Process_Layout;
