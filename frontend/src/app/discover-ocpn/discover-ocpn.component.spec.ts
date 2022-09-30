import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DiscoverOcpnComponent } from './discover-ocpn.component';

describe('DiscoverOcpnComponent', () => {
  let component: DiscoverOcpnComponent;
  let fixture: ComponentFixture<DiscoverOcpnComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DiscoverOcpnComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DiscoverOcpnComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
