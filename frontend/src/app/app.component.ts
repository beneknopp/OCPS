import { Component } from '@angular/core';
import { AppService } from './app.service';
import { DOMService } from './dom.service';
import { faAnglesRight } from '@fortawesome/free-solid-svg-icons';
import { Router } from '@angular/router';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  
  title = 'OCPSim';
  ping: String = "";
  fileName = "";
  ocel_info  = {
    "otypes" : [],
    "acts": [],
    "activity-allowed-otypes" : {},
    "activity-leading-otype-candidates" : {},
    flag: false
  };
  ocel_initialized = false;
  doubleRightArrowsIcon = faAnglesRight;

  constructor(
    private appService: AppService,
    public domService: DOMService,
    public router: Router
  ) {}

  getPing(): void {
    this.appService.getPing().subscribe(resp => {
      console.log(resp)
      this.ping = resp
    });
  }

}
