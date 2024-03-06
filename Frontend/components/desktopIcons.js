import React from "react";
import IconBox from "./iconBox";
import IconText from "./iconText";
// import "./styles.scss";
import { Awfxcg321303, Explorer103, CdMusic , CdSearch ,Cdplayer110} from "@react95/icons";

function DesktopIcon({ openPortfolio, openCV, openTunes ,openSC_Manager, openSC_Process}) {
  return (
    <div>
        <IconBox className="pointer" onClick={() => openPortfolio()}>
          <Explorer103 className="pointer" variant="32x32_4" />
          <IconText className="pointer">Portfolio.txt</IconText>
        </IconBox>
        <IconBox className="pointer" onClick={() => openCV()}>
          <Awfxcg321303 className="pointer" variant="32x32_4" />
          <IconText className="pointer">CV.txt</IconText>
        </IconBox>
        <IconBox className="pointer" onClick={() => openTunes()}>
          <CdMusic className="pointer" variant="32x32_4" />
          <IconText className="pointer">Tunes</IconText>
        </IconBox>
        <IconBox className="pointer" >
          <CdSearch className="pointer" variant="32x32_4" />
          <IconText className="pointer">Q & A_SCV</IconText>
        </IconBox>
        <IconBox className="pointer" onClick={() => openSC_Process()} >
          <Cdplayer110 className="pointer" variant="32x32_4" />
          <IconText className="pointer">SC_Process</IconText>
        </IconBox>
        <IconBox className="pointer" onClick={() => openSC_Manager()} >
          <CdMusic className="pointer" variant="32x32_4" />
          <IconText className="pointer">SC_Manager</IconText>
        </IconBox>
    </div>
  );
}

export default DesktopIcon;
